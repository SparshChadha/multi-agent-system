# classifier_agent.py
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("GEMINI_KEY")

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Changed from gemini-pro as gemini-2.0-flash is current
    temperature=0,
    # max_tokens=None, # Typically not needed for ChatGoogleGenerativeAI, remove if problematic
    # timeout=None, # Typically not needed, remove if problematic
    max_retries=2,
    # Ensure API key is set as an environment variable: GOOGLE_API_KEY
    api_key=api_key # Hardcoding API keys is not recommended. Prefer environment variables.
)
template = """
You are an agent—please keep going until the user's query is completely resolved. Only terminate your turn when you are sure you have correctly classified the input.
If you are not sure whether the input is JSON, Email, or PDF, use any available tools to inspect file names, extensions, or content. Do NOT guess.
You must classify both:
1. Format (one of JSON, Email, or PDF)
2. Business Intent (one of RFQ, Complaint, Invoice, Regulation, or Fraud Risk)

Return your answer EXACTLY as:
<FORMAT>|<INTENT>
with no extra words, punctuation, or explanation.
For example:
Email|Complaint

If you cannot confidently identify format or intent, respond with:
UNKNOWN|UNKNOWN

---

# Instructions

1. Parse the entire input carefully.
   - If it looks like a JSON payload or a webhook body (e.g. starts with "{{" or "[{{" or contains clear ""key"":""value"" syntax), classify as JSON.
   - If it has typical email headers (From:, To:, Subject:) or looks like a plain-text email body, classify as Email.
   - If it contains natural-language paragraphs plus references to invoice line-items (e.g. “Invoice #1234”, “Total Due: $…”]) or legal-policy language (“This policy states…”, “GDPR”), classify as PDF.
   - If you are using any provided tools (e.g. get_weather, get_current_time), you MAY call them, but only if it helps you verify “format” is JSON vs. Email vs. PDF. Otherwise, rely on content clues.

2. Once you know the format, identify business intent from the main body text:
   - RFQ (Request for Quotation) if it talks about “pricing,” “quote,” “we'd like to purchase,” “requesting a quote for X units.”
   - Complaint if it describes an issue (“broken,” “missing,” “damaged,” “unacceptable service,” “I want a refund,” “I'm upset”, "ASAP", "immediately", "unacceptable").
   - Invoice if it is an invoice document (table of line items, “Invoice Number,” “Amount Due,” “Bill To,” “Due Date”, "Subtotal", "Total Due").
   - Regulation if it is a legal/policy document (mentions “GDPR,” “FDA,” “HIPAA,” “compliance,” “section,” “clause,” “regulation”, "policy").
   - Fraud Risk if it explicitly calls out “fraud,” “suspicious transaction,” “unauthorized,” “chargeback risk,” “identity theft,” or “phishing alert”, "security alert".

3. Enforce strict formatting:
   - ALWAYS output exactly two tokens separated by a single pipe (|), e.g. Email|RFQ.
   - Do not include additional text, punctuation, or explanation.
   - Do not lowercase or uppercase incorrectly: JSON, Email, PDF; RFQ, Complaint, Invoice, Regulation, Fraud Risk.

---

# Output Format

<FORMAT>|<INTENT>
- <FORMAT> must be one of: JSON, Email, PDF
- <INTENT> must be one of: RFQ, Complaint, Invoice, Regulation, Fraud Risk

If uncertain, output:
UNKNOWN|UNKNOWN

---

# Examples

### Example 1
Input (Email-style text):
From: alice@example.com
To: support@flowbit.com
Subject: Missing Widget in Order #1234

Hello Flowbit Team,

I just received my order yesterday, but the widget I paid for is missing from the package. I need a replacement ASAP. This is unacceptable.

Thanks,
Alice Johnson

Output:
Email|Complaint

### Example 2
Input (JSON payload):
{{
  "event": "order_placed",
  "order_id": "ORD-5678",
  "amount": 1250.50,
  "currency": "USD",
  "customer_id": "CUST-9012"
}}

Output:
JSON|RFQ

### Example 3
Input (Invoice-style PDF text snippet):
INVOICE
Invoice Number: INV-2025-0042
Bill To: Acme Corp.
Date: 2025-05-30

Description          Qty   Unit Price   Line Total
---------------------------------------------------
Consulting Services   10     200.00       2000.00
Hardware Components    5     500.00       2500.00

                     Subtotal: 4500.00
                     Tax (10%):  450.00
                     Total Due:  4950.00

Output:
PDF|Invoice

### Example 4
Input (Policy document fragment):
FLOWBIT DATA PROTECTION POLICY
Effective Date: June 1, 2025

1. Introduction
Flowbit must comply with GDPR (General Data Protection Regulation) requirements. Any data transfer outside the EU must meet Article 45. ...

Output:
PDF|Regulation

### Example 5
Input (Email with potential fraud mention):
From: fraud-alert@bank.com
To: security@flowbit.com
Subject: Suspicious Charge Detected

We have detected a suspicious transaction on your account center last night. It looks like an unauthorized $10,000 transfer. Please investigate immediately to mitigate any fraud risk.

Output:
Email|Fraud Risk

---

Now, classify the following input accordingly:
{user_input}
"""

# Create the PromptTemplate
prompt = PromptTemplate(
    input_variables=["user_input"],
    template=template,
)

# Create the chain using the RunnableSequence syntax (prompt | llm)
# The output of this chain will typically be an AIMessage, and you'll access its content.
# For LangGraph, often we need the direct string or dict output.
# We'll wrap it slightly to make it behave more like the old LLMChain for the orchestration script if needed,
# or adjust the orchestration script to handle AIMessage.
# For now, let's assume the orchestration script expects a dict with a "text" key like the old LLMChain.run
# A simple way to achieve similar output structure for this specific case:
chain = prompt | llm | (lambda ai_message: {"text": ai_message.content})


def classify_input(text: str) -> str:
    """
    Runs the chain on the given text and returns the "<FORMAT>|<INTENT>" result.
    """
    # The wrapped chain now returns a dict like {"text": "Format|Intent"}
    result_dict = chain.invoke({"user_input": text})
    return result_dict["text"].strip()


def make_chain(): # Removed type hint -> LLMChain
    """
    Creates and returns a RunnableSequence for classification.
    This is the function that orchestration.py expects to import.
    """
    return chain
