from langgraph.graph import Graph
from langgraph.graph.graph import END
import json
import time
import redis # Import the redis library
import uuid # For more robust unique IDs
import sys
from pathlib import Path
import argparse
import mimetypes # For detecting file types
from tika import parser as tika_parser # For PDF text extraction
from faker import Faker # For generating fake data
import os # For temp file operations

# Ensure the script can find core_Agents
sys.path.append(str(Path(__file__).parent.parent))

# Assume the user has four modules, each exposing make_chain() returning an LLMChain-like RunnableSequence:
from core_agents.classifier_agent import make_chain as make_classifier_chain
from core_agents.email_agent import make_chain as make_email_chain
from core_agents.json_agent import make_chain as make_json_chain
from core_agents.pdf_agent import make_chain as make_pdf_chain


# =======================================================================
# Shared Memory Store (Redis Configuration)
# =======================================================================
redis_client = None
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("[MEMORY] Successfully connected to Redis!")
except redis.exceptions.ConnectionError as e:
    print(f"[MEMORY_ERROR] Could not connect to Redis: {e}")
    print("[MEMORY_INFO] Proceeding without Redis. Memory will not be saved.")
except Exception as e:
    print(f"[MEMORY_ERROR] An unexpected error occurred during Redis connection: {e}")
    print("[MEMORY_INFO] Proceeding without Redis. Memory will not be saved.")


def write_to_memory(orchestration_id: str, item_key: str, item_data: dict):
    if not redis_client:
        return

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    record = {"timestamp": timestamp}
    for k, v in item_data.items():
        if isinstance(v, (dict, list)):
            record[k] = json.dumps(v)
        else:
            record[k] = str(v) # Ensure all values are strings for hmset
    
    entry_id = f"{item_key}:{time.time()}"
    redis_hash_key = f"orchestration:{orchestration_id}:entry:{entry_id}"
    
    try:
        redis_client.hmset(redis_hash_key, record)
        redis_client.zadd(f"orchestration_log:{orchestration_id}", {redis_hash_key: time.time()})
        # print(f"[MEMORY_REDIS] {timestamp} - Orchestration {orchestration_id} - Key {item_key}: {item_data}")
    except redis.exceptions.RedisError as e:
        print(f"[MEMORY_REDIS_ERROR] Could not write to Redis for key {redis_hash_key}: {e}")
    except Exception as e:
        print(f"[MEMORY_REDIS_ERROR] An unexpected error occurred while writing to Redis for key {redis_hash_key}: {e}")


def get_orchestration_log(orchestration_id: str) -> list:
    if not redis_client:
        return []
        
    try:
        entry_keys = redis_client.zrange(f"orchestration_log:{orchestration_id}", 0, -1)
        log = []
        for key in entry_keys:
            entry_data = redis_client.hgetall(key)
            if entry_data:
                for k, v_str in entry_data.items():
                    try:
                        # Attempt to parse if it looks like JSON, otherwise keep as string
                        if (v_str.startswith('{') and v_str.endswith('}')) or \
                           (v_str.startswith('[') and v_str.endswith(']')):
                            entry_data[k] = json.loads(v_str)
                        else:
                            entry_data[k] = v_str # Keep as string if not obviously JSON
                    except json.JSONDecodeError:
                        entry_data[k] = v_str # Keep original string if parsing fails
                log.append(entry_data)
        return log
    except redis.exceptions.RedisError as e:
        print(f"[MEMORY_REDIS_ERROR] Could not read orchestration log from Redis for {orchestration_id}: {e}")
        return []
    except Exception as e:
        print(f"[MEMORY_REDIS_ERROR] An unexpected error occurred while reading orchestration log from Redis for {orchestration_id}: {e}")
        return []

# =======================================================================
# PDF and Input Processing
# =======================================================================
def extract_text_from_pdf_path_with_tika(pdf_path: str) -> str | None:
    """Extracts text from a PDF file using Apache Tika."""
    try:
        # Ensure Tika server is running (e.g., via Docker: docker run -d -p 9998:9998 apache/tika:latest)
        # The tika-python client defaults to http://localhost:9998/
        print(f"[TIKA_INFO] Attempting to parse PDF: {pdf_path}")
        raw = tika_parser.from_file(pdf_path)
        if raw and 'content' in raw and raw['content']:
            print(f"[TIKA_SUCCESS] Successfully extracted text from {pdf_path}")
            return raw['content'].strip()
        else:
            print(f"[TIKA_WARN] No content extracted from {pdf_path}. Raw Tika output: {raw}")
            return None
    except ConnectionRefusedError:
        print("[TIKA_ERROR] Connection to Tika server refused. Is the Tika server running on port 9998?")
        return None
    except Exception as e:
        print(f"[TIKA_ERROR] Failed to extract text from PDF {pdf_path} using Tika: {e}")
        return None

def preprocess_input_for_agents(input_data: any, input_type_hint: str = "unknown") -> tuple[str, str]:
    """
    Processes input data (text, JSON dict, PDF path) into a standardized string for the classifier.
    Returns a tuple: (processed_string_for_classifier, original_content_for_agent).
    The original_content_for_agent will be the raw text (for email/PDF) or JSON string.
    """
    original_content = ""
    processed_text_for_classifier = ""
    data_type = "unknown"

    if isinstance(input_data, str):
        # Could be raw text (email), JSON string, or path to a PDF
        if input_type_hint == "pdf" or (input_data.lower().endswith(".pdf") and Path(input_data).is_file()):
            data_type = "pdf"
            pdf_text = extract_text_from_pdf_path_with_tika(input_data)
            if pdf_text:
                original_content = pdf_text
                processed_text_for_classifier = f"type: pdf\nfilepath: {input_data}\ncontent:\n{pdf_text[:2000]}" # Truncate for classifier
            else:
                original_content = f"Error: Could not extract text from PDF {input_data}"
                processed_text_for_classifier = f"type: pdf\nfilepath: {input_data}\ncontent: Error processing PDF."
        elif (input_data.strip().startswith("{") and input_data.strip().endswith("}")) or \
             (input_data.strip().startswith("[") and input_data.strip().endswith("]")):
            data_type = "json"
            original_content = input_data # Keep original JSON string for JSON agent
            try:
                json_obj = json.loads(input_data)
                # Basic heuristic for "subject" from common event fields
                subject_heuristic = json_obj.get("event", json_obj.get("event_type", json_obj.get("message_type", "JSON Data")))
                processed_text_for_classifier = f"type: json\nsubject: {subject_heuristic}\ncontent:\n{input_data[:2000]}"
            except json.JSONDecodeError:
                processed_text_for_classifier = f"type: json\ncontent: Invalid JSON string provided\n{input_data[:2000]}"
        else: # Assume it's email-like text
            data_type = "email"
            original_content = input_data # Keep original for email agent
            # Basic heuristic for email fields for classifier
            lines = input_data.splitlines()
            sender_heuristic = next((line.split(":",1)[1].strip() for line in lines if line.lower().startswith("from:")), "unknown_sender")
            to_heuristic = next((line.split(":",1)[1].strip() for line in lines if line.lower().startswith("to:")), "unknown_recipient")
            subject_heuristic = next((line.split(":",1)[1].strip() for line in lines if line.lower().startswith("subject:")), "No Subject")
            processed_text_for_classifier = f"type: email\nsender: {sender_heuristic}\nto: {to_heuristic}\nsubject: {subject_heuristic}\ncontent:\n{input_data[:2000]}"

    elif isinstance(input_data, dict): # Assume it's pre-parsed JSON
        data_type = "json"
        original_content = json.dumps(input_data, indent=2) # For JSON agent
        subject_heuristic = input_data.get("event", input_data.get("event_type", input_data.get("message_type", "JSON Data")))
        processed_text_for_classifier = f"type: json\nsubject: {subject_heuristic}\ncontent:\n{json.dumps(input_data)[:2000]}"
    else:
        original_content = f"Error: Unsupported input type: {type(input_data)}"
        processed_text_for_classifier = f"type: unknown\ncontent: Unsupported input type provided."

    print(f"[PREPROCESS] Detected input type: {data_type}")
    # print(f"[PREPROCESS] Text for Classifier: {processed_text_for_classifier[:300]}...")
    # print(f"[PREPROCESS] Original Content for Agent (start): {original_content[:300]}...")
    return processed_text_for_classifier, original_content

# =======================================================================
# Action Router function
# =======================================================================
def action_router(orchestration_id: str, agent_name: str, agent_output: dict):
    action_details = {"agent_name": agent_name, "agent_output": agent_output, "triggered_action": "NO_MATCHING_ACTION_RULE"}

    if not isinstance(agent_output, dict):
        print(f"[ACTION_ERROR] agent_output for {agent_name} is not a dictionary: {agent_output}")
        action_details["triggered_action"] = "ERROR: agent_output was not a dictionary"
        write_to_memory(orchestration_id, "action_routing_error", action_details)
        return

    # (Action router logic remains the same as your provided script)
    if agent_name == "EmailAgent":
        if agent_output.get("spam_flag") == "PHISHING_RISK":
            action_details["triggered_action"] = f"POST /alerts/email_phishing with details: {agent_output.get('sender')}, {agent_output.get('issue_summary')}"
        elif agent_output.get("department") == "billing":
            action_details["triggered_action"] = f"FORWARD to Billing Team: {agent_output.get('issue_summary')} from {agent_output.get('sender')}"
        elif agent_output.get("action") == "ESCALATE":
            action_details["triggered_action"] = f"POST /crm/escalate with payload {agent_output}"
        elif agent_output.get("action") == "ACKNOWLEDGE":
            action_details["triggered_action"] = f"POST /email/send_acknowledgment to {agent_output.get('sender')} regarding '{agent_output.get('issue_summary')}'"
        elif agent_output.get("action") == "ROUTINE":
             action_details["triggered_action"] = "LOG and CLOSE (email was routine)"
        else:
            action_details["triggered_action"] = f"LOG and CLOSE (email, unhandled action: {agent_output.get('action', 'N/A')})"

    elif agent_name == "JSONAgent":
        status = agent_output.get("validation_status")
        if status == "INVALID_JSON":
            action_details["triggered_action"] = f"POST /alerts/json_parse_error with errors={agent_output.get('errors')}"
        elif status == "INVALID_SCHEMA":
            action_details["triggered_action"] = f"POST /alerts/json_schema_error with errors={agent_output.get('errors')}, data={agent_output.get('data')}"
        elif status == "DEPRECATED_SCHEMA":
            action_details["triggered_action"] = f"SEND notice: upgrade to new webhook schema for {agent_output.get('data', {}).get('event', 'unknown event')}"
        elif status == "UNKNOWN_EVENT":
            action_details["triggered_action"] = f"LOG unexpected event type: {agent_output.get('data', {}).get('event')} from data: {agent_output.get('data')}"
        elif status == "VALID":
            action_details["triggered_action"] = f"STORE data in database for event: {agent_output.get('data', {}).get('event')}"
        else:
            action_details["triggered_action"] = f"UNKNOWN JSON status: {status}, errors: {agent_output.get('errors')}"

    elif agent_name == "PDFAgent":
        inv_flag = agent_output.get("invoice_flag")
        pol_flag = agent_output.get("policy_flag")
        pdf_type = agent_output.get("pdf_type")

        if pdf_type == "INVOICE":
            if inv_flag == "HIGH_VALUE_INVOICE":
                total = agent_output.get("total")
                action_details["triggered_action"] = f"POST /risk_alert/invoice with total={total}, invoice_number={agent_output.get('invoice_number')}"
            elif inv_flag == "DUPLICATE_INVOICE":
                inv_no = agent_output.get("invoice_number")
                action_details["triggered_action"] = f"POST /alerts/duplicate_invoice with invoice_number={inv_no}"
            elif inv_flag == "INCOMPLETE_INVOICE":
                action_details["triggered_action"] = f"FLAG invoice {agent_output.get('invoice_number')} as incomplete and request missing fields"
            elif inv_flag == "PAST_DUE_INVOICE":
                action_details["triggered_action"] = f"POST /billing/notify_late_payment for invoice {agent_output.get('invoice_number')}"
            elif inv_flag == "NONE" or inv_flag is None:
                 action_details["triggered_action"] = f"PROCESS invoice {agent_output.get('invoice_number')} (standard)"
            else:
                action_details["triggered_action"] = f"PROCESS invoice {agent_output.get('invoice_number')} (flag: {inv_flag})"
        elif pdf_type == "POLICY":
            if pol_flag == "COMPLIANCE_RISK":
                terms = agent_output.get("key_terms", [])
                action_details["triggered_action"] = f"POST /risk_alert/compliance with terms={terms} for policy '{agent_output.get('policy_title')}'"
            elif pol_flag == "LEGAL_REVIEW_REQUIRED":
                action_details["triggered_action"] = f"NOTIFY legal team for review of policy '{agent_output.get('policy_title')}'"
            elif pol_flag == "NONE" or pol_flag is None:
                 action_details["triggered_action"] = f"ARCHIVE policy document '{agent_output.get('policy_title')}' (standard)"
            else:
                action_details["triggered_action"] = f"PROCESS policy {agent_output.get('policy_title')} (flag: {pol_flag})"
        
        if action_details["triggered_action"] == "NO_MATCHING_ACTION_RULE" and pdf_type not in ["INVOICE", "POLICY"]:
             action_details["triggered_action"] = f"LOG PDF content (type: {pdf_type}, inv_flag: {inv_flag}, pol_flag: {pol_flag})"

    print(f"[ACTION] → Orchestration {orchestration_id}: {action_details['triggered_action']}")
    write_to_memory(orchestration_id, "action_routing", action_details)

# =======================================================================
# Node Definitions
# =======================================================================
def classifier_node_func(inputs):
    """
    Takes 'raw_input_for_classifier' and 'original_content_for_agent' from the preprocessor.
    Classifies 'raw_input_for_classifier'.
    Passes 'original_content_for_agent' along as 'payload' for the next agent.
    """
    raw_text_for_classifier = inputs.get("raw_input_for_classifier")
    original_content = inputs.get("original_content_for_agent") # This is what the specialized agents will use
    orchestration_id = inputs.get("orchestration_id")
    
    chain = make_classifier_chain()
    response = chain.invoke({"user_input": raw_text_for_classifier})
    classification = response["text"].strip() if isinstance(response, dict) and "text" in response else str(response).strip()

    if "|" not in classification:
        print(f"[CLASSIFIER_ERROR] Invalid classification output: {classification}")
        fmt, intent = "UNKNOWN", "UNKNOWN"
    else:
        fmt, intent = classification.split("|", 1)

    # The 'payload' for the router should be the original, unprocessed content
    out = {"format": fmt, "intent": intent, "payload": original_content}
    write_to_memory(orchestration_id, "classification", {"input_classifier": raw_text_for_classifier, **out})
    return {"orchestration_id": orchestration_id, **out}


def router_node_func(inputs):
    fmt = inputs.get("format")
    # Payload is now the original content passed from the classifier node
    payload = inputs.get("payload") 
    orchestration_id = inputs.get("orchestration_id")
    chosen_agent = fmt
    write_to_memory(orchestration_id, "routing", {"chosen_agent": chosen_agent})
    # Pass the original payload to the chosen agent
    return {"orchestration_id": orchestration_id, "chosen_agent": chosen_agent, "payload": payload, "format": fmt, "intent": inputs.get("intent")}

def parse_agent_llm_output(llm_content_str: str) -> dict:
    content = llm_content_str.strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"): lines = lines[1:]
        content = "\n".join(lines).strip()
    
    if content.endswith("```"):
        content = content[:-len("```")].strip()
        
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[PARSE_ERROR] Could not parse LLM output as JSON: {e}\nContent: '{content}'")
        return {"error": "JSONDecodeError", "raw_content": content, "parsing_error_message": str(e)}


def email_agent_node_func(inputs):
    text = inputs.get("payload") # This is the original email text
    orchestration_id = inputs.get("orchestration_id")
    chain = make_email_chain()
    # The email agent expects the raw email text.
    response = chain.invoke({"user_input": text}) 
    parsed = parse_agent_llm_output(response.content if hasattr(response, 'content') else str(response))
    write_to_memory(orchestration_id, "agent_output_EmailAgent", {"agent": "EmailAgent", "output": parsed, "extracted_fields": parsed})
    return {"orchestration_id": orchestration_id, "agent_name": "EmailAgent", "agent_output": parsed}


def json_agent_node_func(inputs):
    text = inputs.get("payload") # This is the original JSON string
    orchestration_id = inputs.get("orchestration_id")
    chain = make_json_chain()
    # The JSON agent expects the raw JSON string.
    response = chain.invoke({"user_input": text}) 
    parsed = parse_agent_llm_output(response.content if hasattr(response, 'content') else str(response))
    extracted_data = None
    if isinstance(parsed, dict) and parsed.get("validation_status") == "VALID":
        extracted_data = parsed.get("data")
    write_to_memory(orchestration_id, "agent_output_JSONAgent", {"agent": "JSONAgent", "output": parsed, "extracted_fields": extracted_data})
    return {"orchestration_id": orchestration_id, "agent_name": "JSONAgent", "agent_output": parsed}


def pdf_agent_node_func(inputs):
    text = inputs.get("payload") # This is the Tika-extracted text from the PDF
    orchestration_id = inputs.get("orchestration_id")
    chain = make_pdf_chain()
    # The PDF agent expects the extracted text.
    # Ensure current_date is passed if your pdf_agent requires it.
    from core_agents.pdf_agent import get_current_date_for_prompt # Assuming this is in your pdf_agent.py
    current_date_str = get_current_date_for_prompt()
    response = chain.invoke({"user_input": text, "current_date": current_date_str})
    parsed = parse_agent_llm_output(response.content if hasattr(response, 'content') else str(response))
    extracted_fields = {}
    if isinstance(parsed, dict):
        extracted_fields = {k: v for k, v in parsed.items() if v is not None and k not in ["pdf_type", "invoice_flag", "policy_flag"]}
    write_to_memory(orchestration_id, "agent_output_PDFAgent", {"agent": "PDFAgent", "output": parsed, "extracted_fields": extracted_fields})
    return {"orchestration_id": orchestration_id, "agent_name": "PDFAgent", "agent_output": parsed}

def action_router_node_func(inputs):
    agent_name = inputs.get("agent_name")
    agent_output = inputs.get("agent_output")
    orchestration_id = inputs.get("orchestration_id")
    action_router(orchestration_id, agent_name, agent_output)
    return {"orchestration_id": orchestration_id, "final_action_details": f"Action routing complete for {agent_name}"}

# =======================================================================
# Build the Graph and connect nodes
# =======================================================================
workflow = Graph()
workflow.add_node("Classifier", classifier_node_func)
workflow.add_node("Router", router_node_func)
workflow.add_node("EmailAgent", email_agent_node_func)
workflow.add_node("JSONAgent", json_agent_node_func)
workflow.add_node("PDFAgent", pdf_agent_node_func)
workflow.add_node("ActionRouter", action_router_node_func)

workflow.add_edge("Classifier", "Router")

def route_logic(inputs):
    if inputs.get("chosen_agent") == "Email":
        return "EmailAgent"
    elif inputs.get("chosen_agent") == "JSON":
        return "JSONAgent"
    elif inputs.get("chosen_agent") == "PDF":
        return "PDFAgent"
    else:
        print(f"[ROUTER_ERROR] Unknown format for routing: {inputs.get('format')}")
        return END

workflow.add_conditional_edges(
    "Router",
    route_logic,
    {
        "EmailAgent": "EmailAgent",
        "JSONAgent": "JSONAgent",
        "PDFAgent": "PDFAgent",
        END: END
    }
)

workflow.add_edge("EmailAgent", "ActionRouter")
workflow.add_edge("JSONAgent",  "ActionRouter")
workflow.add_edge("PDFAgent",   "ActionRouter")

workflow.set_entry_point("Classifier")
workflow.add_edge("ActionRouter", END)

app = workflow.compile()

# =======================================================================
# Faker and Example Generation
# =======================================================================
fake = Faker()

def generate_fake_json_webhook():
    """Generates a fake JSON webhook payload using Faker."""
    event_types = ["order_placed", "user_updated", "payment_failed", "item_shipped", "new_registration"]
    payload_versions = ["v1", "v2", "v3"] # v1 for deprecated example
    data = {
        "event": fake.random_element(elements=event_types),
        "payload_version": fake.random_element(elements=payload_versions),
        "timestamp": fake.iso8601(),
        "transaction_id": fake.uuid4(),
    }
    if "order" in data["event"] or "payment" in data["event"]:
        data["order_id"] = f"ORD-{fake.random_number(digits=5)}"
        data["amount"] = round(fake.pyfloat(left_digits=3, right_digits=2, positive=True), 2)
        data["currency"] = fake.currency_code()
    if "user" in data["event"] or "registration" in data["event"]:
        data["customer_id"] = f"CUST-{fake.random_number(digits=4)}"
        data["user_details"] = {
            "name": fake.name(),
            "email": fake.email(),
            "ip_address": fake.ipv4()
        }
    return json.dumps(data, indent=2)

def generate_fake_email_text():
    """Generates fake email text using Faker."""
    sender = fake.email()
    recipient = fake.email()
    subject_keywords = ["Inquiry", "Urgent Request", "Feedback", "Issue Report", "Question"]
    subject = f"{fake.random_element(elements=subject_keywords)} about {fake.bs()}"
    
    body_intro = [
        f"Dear {fake.name()},\n\nI am writing to you regarding ",
        f"Hello Team,\n\nThis email concerns ",
        f"To Whom It May Concern,\n\nI would like to discuss "
    ]
    body_main_phrases = [
        "the recent order # {fake.random_number(digits=5)}.",
        "a problem I encountered with your service: {fake.sentence(nb_words=10)}.",
        "an urgent matter that requires your immediate attention: {fake.sentence(nb_words=15)}.",
        "a quick question about {fake.word()}.",
        "the invoice {fake.random_number(digits=4)} which appears to have an error."
    ]
    body_closing = [
        "\n\nThanks,\n" + fake.name(),
        "\n\nSincerely,\n" + fake.name(),
        "\n\nPlease look into this ASAP.\nRegards,\n" + fake.name()
    ]
    
    email_text = f"From: {sender}\nTo: {recipient}\nSubject: {subject}\n\n"
    email_text += fake.random_element(elements=body_intro)
    email_text += fake.random_element(elements=body_main_phrases).format(fake=fake) # Pass fake for interpolation
    email_text += fake.random_element(elements=body_closing)
    return email_text

def generate_fake_pdf_text(pdf_type="invoice"):
    """Generates fake text content that might appear in a PDF (invoice or policy)."""
    if pdf_type == "invoice":
        return f"""
INVOICE
Invoice Number: INV-{fake.random_number(digits=5)}
Date: {fake.date_this_year().strftime('%Y-%m-%d')}
Due Date: {fake.date_between(start_date='today', end_date='+30d').strftime('%Y-%m-%d')}

Bill To:
{fake.company()}
{fake.address().replace(chr(10), ', ')}

Item Description         Quantity  Unit Price    Total
---------------------------------------------------------
{fake.bs()}                     {fake.random_int(min=1, max=5)}    {round(fake.pyfloat(2,2,positive=True),2)}     {round(fake.pyfloat(3,2,positive=True),2)}
{fake.bs()}                     {fake.random_int(min=1, max=3)}    {round(fake.pyfloat(2,2,positive=True),2)}     {round(fake.pyfloat(3,2,positive=True),2)}

Subtotal: {round(fake.pyfloat(3,2,positive=True),2)}
Tax (10%): {round(fake.pyfloat(2,2,positive=True),2)}
Total Due: {round(fake.pyfloat(4,2,positive=True),2)}
Please make payments to {fake.company_email()}.
"""
    elif pdf_type == "policy":
        key_terms = fake.random_elements(elements=("GDPR", "Data Security", "Confidentiality", "Compliance", "Internal Audit", "Access Control"), length=fake.random_int(min=1, max=3), unique=True)
        return f"""
{fake.company()} INTERNAL POLICY DOCUMENT
Policy Title: {fake.catch_phrase()} Policy
Effective Date: {fake.date_between(start_date='-30d', end_date='today').strftime('%Y-%m-%d')}
Version: {fake.random_int(min=1,max=3)}.{fake.random_int(min=0,max=9)}

1. Introduction
This document outlines the company's policy regarding {fake.bs()}. Adherence to this policy is mandatory for all employees.
This policy requires legal review due to new regulations.

2. Scope
This policy applies to {fake.paragraph(nb_sentences=2)}.

3. Key Terms
Key terms associated with this policy include: {', '.join(key_terms)}.

4. Procedures
{fake.paragraph(nb_sentences=3)}

5. Non-Compliance
Failure to comply with this policy may result in disciplinary action, up to and including termination of employment. This policy is subject to legal review.
Contact {fake.name()} at {fake.email()} for questions.
"""
    return "Unsupported fake PDF type requested."


# =======================================================================
# Run the graph
# =======================================================================
def run_graph(input_data: any, input_type_hint: str = "unknown"):
    """
    Processes the input data (which can be text, JSON string, dict, or PDF path),
    then runs the graph.
    """
    short_uuid = str(uuid.uuid4())[:8]
    orchestration_id = f"run_{int(time.time())}_{short_uuid}"
    
    print(f"\n===== STARTING ORCHESTRATION {orchestration_id} =====")
    
    # Preprocess the input to get a string for the classifier and original content for agents
    raw_input_for_classifier, original_content_for_agent = preprocess_input_for_agents(input_data, input_type_hint)
    
    print(f"Input for Classifier (first 100 chars): {raw_input_for_classifier[:100]}...")
    print(f"Original Content for Agent (first 100 chars): {original_content_for_agent[:100]}...")

    # The initial payload for the graph now includes both versions of the input
    initial_payload = {
        "raw_input_for_classifier": raw_input_for_classifier,
        "original_content_for_agent": original_content_for_agent,
        "orchestration_id": orchestration_id
    }
    
    try:
        final_state = app.invoke(initial_payload)
        print(f"===== ORCHESTRATION {orchestration_id} COMPLETE =====")
        # print("Final state from app.invoke:", final_state) # Often very verbose
        
        if redis_client:
            full_log = get_orchestration_log(orchestration_id)
            print(f"\n--- Full Log for Orchestration {orchestration_id} from Redis ---")
            if full_log:
                for entry_index, entry in enumerate(full_log):
                    print(f"Log Entry {entry_index + 1}: {json.dumps(entry, indent=2)}") # Pretty print JSON
            else:
                print("No log entries found in Redis for this orchestration.")
            print("--- End of Log ---")

    except Exception as e:
        print(f"[ERROR] Orchestration {orchestration_id} failed: {e}")
        import traceback
        traceback.print_exc()
        if redis_client:
            write_to_memory(orchestration_id, "orchestration_error", {"error": str(e), "traceback": traceback.format_exc()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangGraph workflow with specified input.")
    parser.add_argument("--input", type=str, help="Raw input text, JSON string, or path to a PDF file. If not provided, runs example set.")
    parser.add_argument("--type", type=str, choices=['email', 'json', 'pdf', 'auto'], default='auto', help="Hint for the input type if it's a string (e.g., if providing a PDF path as a string). 'auto' will try to guess.")
    parser.add_argument("--faker", action="store_true", help="Run with Faker-generated examples instead of predefined ones.")
    args = parser.parse_args()

    # Ensure the 'temp_pdfs' directory exists for PDF examples
    TEMP_PDF_DIR = Path(__file__).resolve().parent / "temp_pdfs"
    TEMP_PDF_DIR.mkdir(exist_ok=True)

    if args.input:
        input_type_hint = args.type
        if args.type == 'auto' and isinstance(args.input, str):
            if args.input.lower().endswith(".pdf"):
                input_type_hint = 'pdf'
            elif (args.input.strip().startswith("{") and args.input.strip().endswith("}")) or \
                 (args.input.strip().startswith("[") and args.input.strip().endswith("]")):
                input_type_hint = 'json'
            else:
                input_type_hint = 'email' # Default assumption for other strings
        
        print(f"Received input via command line (type hint: {input_type_hint}). Length: {len(args.input)}")
        run_graph(args.input, input_type_hint=input_type_hint)
    elif args.faker:
        print("\n--- Running with Faker-generated examples ---")
        fake_examples = [
            (generate_fake_email_text(), "email"),
            (generate_fake_json_webhook(), "json"),
            (generate_fake_pdf_text(pdf_type="invoice"), "pdf_text"), # Will be treated as raw text for PDF agent
            (generate_fake_pdf_text(pdf_type="policy"), "pdf_text"),
        ]
         # For a true PDF file example with Faker, we'd need to generate a PDF file.
        # This is more involved. For now, we pass the fake PDF *text*.
        # To test Tika with a real PDF, you'd pass a path to an actual .pdf file.

        for i, (data, data_type_hint) in enumerate(fake_examples):
            print(f"\n--- Running Faker Example {i+1} (Type: {data_type_hint}) ---")
            run_graph(data, input_type_hint=data_type_hint) # Pass the hint
            if i < len(fake_examples) - 1:
                time.sleep(1)

    else:
        print("\n--- Running predefined examples ---")
        # (Your existing examples - ensure PDF examples are paths to actual PDF files if you want Tika to process them as files)
        # For demonstration, let's assume you have sample PDF files.
        # If not, Tika won't be used for these unless they are paths.
        # For now, I'll keep your string-based PDF examples, which will be handled by preprocess_input_for_agents as text.
        # To test Tika properly, create sample.pdf and pass its path.

        # Create a dummy PDF for testing if it doesn't exist
        sample_invoice_pdf_path = TEMP_PDF_DIR / "sample_invoice.txt" # Store as .txt as it's text
        sample_policy_pdf_path = TEMP_PDF_DIR / "sample_policy.txt"

        with open(sample_invoice_pdf_path, "w") as f:
            f.write(generate_fake_pdf_text("invoice")) # Using faker to generate content for a "PDF" text file
        with open(sample_policy_pdf_path, "w") as f:
            f.write(generate_fake_pdf_text("policy"))

        examples = [
            # Example 1: Email - Phishing
            ( """From: secure-alert@totally-a-real-bank.com
To: victim@example.com
Subject: URGENT Security Alert - Account Compromised - Click Here NOW!

Dear Customer,
Your account has been flagged for suspicious activity. Click this link http://bit.ly/notareallink to verify your details immediately or your account will be suspended. This is a phishing attempt.

Sincerely,
Bank Security (not really)""", "email"),

            # Example 2: Email - Billing Inquiry
            ( """From: john.doe@company.com
To: support@example.com
Subject: Question about my recent invoice

Hi team,
I have a question regarding invoice #INV7890. Could you please forward this to the billing department? I think there might be an overcharge.

Thanks,
John Doe""", "email"),
            # Example 3: JSON - Valid
            (generate_fake_json_webhook(), "json"), # Using Faker for a valid JSON example
            # Example 4: "PDF" text (Invoice-like)
            # Pass the PATH to the file to trigger Tika if it were a real PDF.
            # Since it's a text file with PDF-like content, we pass its content or path.
            # For tika to parse, it should be a .pdf. For now, passing path to .txt file
            (str(sample_invoice_pdf_path), "pdf"), # Pass path to use Tika (even if it's a .txt file, Tika might try)
            # Example 5: "PDF" text (Policy-like)
            (str(sample_policy_pdf_path), "pdf"),
        ]
        for i, (raw_data, data_type_hint) in enumerate(examples):
            print(f"\n--- Running Example {i+1} (Type hint: {data_type_hint}) ---")
            run_graph(raw_data, input_type_hint=data_type_hint) # Pass the type hint
            if i < len(examples) - 1 :
                time.sleep(1)
