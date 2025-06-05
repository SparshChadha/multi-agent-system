from langgraph.graph import Graph
from langgraph.graph.graph import END
import json
import time
import redis # Import the redis library
import uuid # For more robust unique IDs
import sys
from pathlib import Path
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
        # print(f"[MEMORY_SKIP] Redis not available. Skipping write for Orchestration {orchestration_id}, Key {item_key}: {item_data}")
        return

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    record = {"timestamp": timestamp}
    for k, v in item_data.items():
        if isinstance(v, (dict, list)):
            record[k] = json.dumps(v) 
        else:
            record[k] = v
    
    entry_id = f"{item_key}:{time.time()}" 
    redis_hash_key = f"orchestration:{orchestration_id}:entry:{entry_id}"
    
    try:
        redis_client.hmset(redis_hash_key, record)
        redis_client.zadd(f"orchestration_log:{orchestration_id}", {redis_hash_key: time.time()})
        print(f"[MEMORY_REDIS] {timestamp} - Orchestration {orchestration_id} - Key {item_key}: {item_data}")
    except redis.exceptions.RedisError as e:
        print(f"[MEMORY_REDIS_ERROR] Could not write to Redis for key {redis_hash_key}: {e}")
    except Exception as e:
        print(f"[MEMORY_REDIS_ERROR] An unexpected error occurred while writing to Redis for key {redis_hash_key}: {e}")


def get_orchestration_log(orchestration_id: str) -> list:
    if not redis_client:
        # print(f"[MEMORY_SKIP] Redis not available. Skipping get_orchestration_log for Orchestration {orchestration_id}")
        return []
        
    try:
        entry_keys = redis_client.zrange(f"orchestration_log:{orchestration_id}", 0, -1)
        log = []
        for key in entry_keys:
            entry_data = redis_client.hgetall(key)
            if entry_data:
                for k, v in entry_data.items():
                    if isinstance(v, str) and (v.startswith('{') and v.endswith('}')) or (v.startswith('[') and v.endswith(']')):
                        try:
                            entry_data[k] = json.loads(v)
                        except json.JSONDecodeError:
                            pass 
                log.append(entry_data)
        return log
    except redis.exceptions.RedisError as e:
        print(f"[MEMORY_REDIS_ERROR] Could not read orchestration log from Redis for {orchestration_id}: {e}")
        return []
    except Exception as e:
        print(f"[MEMORY_REDIS_ERROR] An unexpected error occurred while reading orchestration log from Redis for {orchestration_id}: {e}")
        return []

# =======================================================================
# Action Router function (UPDATED with new logic)
# =======================================================================
def action_router(orchestration_id: str, agent_name: str, agent_output: dict):
    action_details = {"agent_name": agent_name, "agent_output": agent_output, "triggered_action": "NO_MATCHING_ACTION_RULE"} # Default action

    if not isinstance(agent_output, dict):
        print(f"[ACTION_ERROR] agent_output for {agent_name} is not a dictionary: {agent_output}")
        action_details["triggered_action"] = "ERROR: agent_output was not a dictionary"
        write_to_memory(orchestration_id, "action_routing_error", action_details)
        return

    if agent_name == "EmailAgent":
        # Order of checks matters here - more specific first
        if agent_output.get("spam_flag") == "PHISHING_RISK":
            action_details["triggered_action"] = f"POST /alerts/email_phishing with details: {agent_output.get('sender')}, {agent_output.get('issue_summary')}"
        elif agent_output.get("department") == "billing":
            action_details["triggered_action"] = f"FORWARD to Billing Team: {agent_output.get('issue_summary')} from {agent_output.get('sender')}"
        elif agent_output.get("action") == "ESCALATE":
            action_details["triggered_action"] = f"POST /crm/escalate with payload {agent_output}"
        elif agent_output.get("action") == "ACKNOWLEDGE": # Ensure EmailAgent can produce this action
            action_details["triggered_action"] = f"POST /email/send_acknowledgment to {agent_output.get('sender')} regarding '{agent_output.get('issue_summary')}'"
        elif agent_output.get("action") == "ROUTINE": # Default action from email agent if no other conditions met
             action_details["triggered_action"] = "LOG and CLOSE (email was routine)"
        else:
            # Fallback if 'action' key exists but isn't handled, or if primary conditions not met
            action_details["triggered_action"] = f"LOG and CLOSE (email, unhandled action: {agent_output.get('action', 'N/A')})"


    elif agent_name == "JSONAgent":
        status = agent_output.get("validation_status")
        if status == "INVALID_JSON":
            action_details["triggered_action"] = f"POST /alerts/json_parse_error with errors={agent_output.get('errors')}"
        elif status == "INVALID_SCHEMA":
            action_details["triggered_action"] = f"POST /alerts/json_schema_error with errors={agent_output.get('errors')}, data={agent_output.get('data')}"
        elif status == "DEPRECATED_SCHEMA": # Ensure JSONAgent can produce this status
            action_details["triggered_action"] = f"SEND notice: upgrade to new webhook schema for {agent_output.get('data', {}).get('event', 'unknown event')}"
        elif status == "UNKNOWN_EVENT": # Ensure JSONAgent can produce this status
            action_details["triggered_action"] = f"LOG unexpected event type: {agent_output.get('data', {}).get('event')} from data: {agent_output.get('data')}"
        elif status == "VALID":
            action_details["triggered_action"] = f"STORE data in database for event: {agent_output.get('data', {}).get('event')}"
        else:
            action_details["triggered_action"] = f"UNKNOWN JSON status: {status}, errors: {agent_output.get('errors')}"

    elif agent_name == "PDFAgent":
        inv_flag = agent_output.get("invoice_flag")
        pol_flag = agent_output.get("policy_flag")
        pdf_type = agent_output.get("pdf_type")

        # Invoice specific flags take precedence if pdf_type is INVOICE
        if pdf_type == "INVOICE":
            if inv_flag == "HIGH_VALUE_INVOICE":
                total = agent_output.get("total")
                action_details["triggered_action"] = f"POST /risk_alert/invoice with total={total}, invoice_number={agent_output.get('invoice_number')}"
            elif inv_flag == "DUPLICATE_INVOICE": # Ensure PDFAgent can produce this
                inv_no = agent_output.get("invoice_number")
                action_details["triggered_action"] = f"POST /alerts/duplicate_invoice with invoice_number={inv_no}"
            elif inv_flag == "INCOMPLETE_INVOICE": # Ensure PDFAgent can produce this
                action_details["triggered_action"] = f"FLAG invoice {agent_output.get('invoice_number')} as incomplete and request missing fields"
            elif inv_flag == "PAST_DUE_INVOICE": # Ensure PDFAgent can produce this
                action_details["triggered_action"] = f"POST /billing/notify_late_payment for invoice {agent_output.get('invoice_number')}"
            # Add a fallback for general invoices if no specific flag is hit but it's an invoice
            elif inv_flag == "NONE" or inv_flag is None: # Assuming "NONE" means no specific alert condition for invoice
                 action_details["triggered_action"] = f"PROCESS invoice {agent_output.get('invoice_number')} (standard)"
            else: # Catch any other invoice flags
                action_details["triggered_action"] = f"PROCESS invoice {agent_output.get('invoice_number')} (flag: {inv_flag})"


        # Policy specific flags, only if not primarily handled by an invoice flag
        # or if pdf_type is POLICY
        if pdf_type == "POLICY": # This condition ensures these only apply if it's a policy
            if pol_flag == "COMPLIANCE_RISK":
                terms = agent_output.get("key_terms", [])
                action_details["triggered_action"] = f"POST /risk_alert/compliance with terms={terms} for policy '{agent_output.get('policy_title')}'"
            elif pol_flag == "LEGAL_REVIEW_REQUIRED": # Ensure PDFAgent can produce this
                action_details["triggered_action"] = f"NOTIFY legal team for review of policy '{agent_output.get('policy_title')}'"
            elif pol_flag == "NONE" or pol_flag is None: # Assuming "NONE" means no specific alert condition for policy
                 action_details["triggered_action"] = f"ARCHIVE policy document '{agent_output.get('policy_title')}' (standard)"
            else: # Catch any other policy flags
                action_details["triggered_action"] = f"PROCESS policy {agent_output.get('policy_title')} (flag: {pol_flag})"
        
        # If pdf_type is unknown or neither invoice nor policy logic applied explicitly
        if action_details["triggered_action"] == "NO_MATCHING_ACTION_RULE" and pdf_type not in ["INVOICE", "POLICY"]:
             action_details["triggered_action"] = f"LOG PDF content (type: {pdf_type}, inv_flag: {inv_flag}, pol_flag: {pol_flag})"


    print(f"[ACTION] → Orchestration {orchestration_id}: {action_details['triggered_action']}")
    write_to_memory(orchestration_id, "action_routing", action_details)

# =======================================================================
# Node Definitions
# =======================================================================
def classifier_node_func(inputs):
    raw_text = inputs.get("raw_input")
    orchestration_id = inputs.get("orchestration_id") 
    chain = make_classifier_chain()
    
    response = chain.invoke({"user_input": raw_text})
    classification = response["text"].strip() if isinstance(response, dict) and "text" in response else str(response).strip()

    if "|" not in classification:
        print(f"[CLASSIFIER_ERROR] Invalid classification output: {classification}")
        fmt, intent = "UNKNOWN", "UNKNOWN"
    else:
        fmt, intent = classification.split("|", 1)

    out = {"format": fmt, "intent": intent, "raw_input": raw_text}
    write_to_memory(orchestration_id, "classification", {"input_source": "upload", **out}) 
    return {"orchestration_id": orchestration_id, **out}


def router_node_func(inputs):
    fmt = inputs.get("format")
    raw_text = inputs.get("raw_input")
    orchestration_id = inputs.get("orchestration_id")
    chosen_agent = fmt
    payload = raw_text
    write_to_memory(orchestration_id, "routing", {"chosen_agent": chosen_agent})
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
    text = inputs.get("payload")
    orchestration_id = inputs.get("orchestration_id")
    chain = make_email_chain()
    response = chain.invoke({"user_input": text})
    parsed = parse_agent_llm_output(response.content if hasattr(response, 'content') else str(response))
    write_to_memory(orchestration_id, "agent_output_EmailAgent", {"agent": "EmailAgent", "output": parsed, "extracted_fields": parsed}) 
    return {"orchestration_id": orchestration_id, "agent_name": "EmailAgent", "agent_output": parsed}


def json_agent_node_func(inputs):
    text = inputs.get("payload")
    orchestration_id = inputs.get("orchestration_id")
    chain = make_json_chain()
    response = chain.invoke({"user_input": text})
    parsed = parse_agent_llm_output(response.content if hasattr(response, 'content') else str(response))
    extracted_data = None
    if isinstance(parsed, dict) and parsed.get("validation_status") == "VALID":
        extracted_data = parsed.get("data")
    write_to_memory(orchestration_id, "agent_output_JSONAgent", {"agent": "JSONAgent", "output": parsed, "extracted_fields": extracted_data})
    return {"orchestration_id": orchestration_id, "agent_name": "JSONAgent", "agent_output": parsed}


def pdf_agent_node_func(inputs):
    text = inputs.get("payload")
    orchestration_id = inputs.get("orchestration_id")
    chain = make_pdf_chain()
    response = chain.invoke({"user_input": text})
    parsed = parse_agent_llm_output(response.content if hasattr(response, 'content') else str(response))
    extracted_fields = {}
    if isinstance(parsed, dict):
        extracted_fields = {k: v for k, v in parsed.items() if v is not None and k not in ["pdf_type", "invoice_flag", "policy_flag"]} # Adjusted to not include flags
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
# Run the graph on some examples
# =======================================================================
def run_graph(raw_input: str):
    short_uuid = str(uuid.uuid4())[:8]
    orchestration_id = f"run_{int(time.time())}_{short_uuid}"
    
    print(f"\n===== STARTING ORCHESTRATION {orchestration_id} =====")
    print(f"Input data: {raw_input[:100]}...") 
    
    initial_payload = {"raw_input": raw_input, "orchestration_id": orchestration_id}
    
    try:
        final_state = app.invoke(initial_payload)
        print(f"===== ORCHESTRATION {orchestration_id} COMPLETE =====")
        print("Final state from app.invoke:", final_state)
        
        if redis_client:
            full_log = get_orchestration_log(orchestration_id)
            print(f"\n--- Full Log for Orchestration {orchestration_id} from Redis ---")
            if full_log:
                for entry_index, entry in enumerate(full_log):
                    print(f"Log Entry {entry_index + 1}: {entry}")
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
    examples = [
        # Example 1: Email - Phishing
        """From: secure-alert@totally-a-real-bank.com
To: victim@example.com
Subject: URGENT Security Alert - Account Compromised - Click Here NOW!

Dear Customer,
Your account has been flagged for suspicious activity. Click this link http://bit.ly/notareallink to verify your details immediately or your account will be suspended. This is a phishing attempt.

Sincerely,
Bank Security (not really)""",

        # Example 2: Email - Billing Inquiry
        """From: john.doe@company.com
To: support@example.com
Subject: Question about my recent invoice

Hi team,
I have a question regarding invoice #INV7890. Could you please forward this to the billing department? I think there might be an overcharge.

Thanks,
John Doe""",

#         # Example 3: Email - Standard Acknowledgment
#         """From: jane.roe@client.com
# To: info@example.com
# Subject: Quick question about your services

# Hello,
# I'd like to know more about your premium service package. No rush, just send details when you can.

# Best,
# Jane Roe""",
        
#         # Example 4: JSON - Deprecated Schema (Illustrative - agent needs to be taught this)
#         """
#             {
#                 "apiVersion": "v1", 
#                 "event_type": "payment_received",
#                 "transaction_id": "txn_old_123",
#                 "amount": 50.00,
#                 "ccy": "USD"
#             }
#         """,

#         # Example 5: JSON - Unknown Event Type
#         """
#             {
#                 "event": "user_played_game", 
#                 "order_id": "GAME-5678",
#                 "amount": 0,
#                 "currency": "POINTS",
#                 "customer_id": "CUST-XYZ"
#             }
#         """,
#         # Example 6: PDF - Incomplete Invoice (Illustrative)
#         """INVOICE
# Invoice Number: INV-2025-0101
# Bill To: Vague Corp.
# Date: 2025-06-01

# Description Qty Unit Price Line Total
# Item A 1 100.00 ?

# Subtotal: ?
# Tax: ?
# Total Due: ?
# This invoice seems to be missing some line item totals and the grand total.
# """,
#         # Example 7: PDF - Legal Review for Policy (Illustrative)
#         """NEW INTERNAL DATA HANDLING POLICY
# Effective Date: July 1, 2025

# This policy outlines new procedures for handling sensitive international client data and requires careful review by our legal department before full implementation due to potential cross-border legal implications. It touches upon GDPR and CCPA.
# Key terms: international data, legal review, GDPR, CCPA.
# """
    ]
    for i, raw in enumerate(examples):
        print(f"\n--- Running Example {i+1} ---")
        run_graph(raw)
        if i < len(examples) - 1 : 
            time.sleep(1)
