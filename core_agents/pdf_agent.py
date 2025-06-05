from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("GEMINI_KEY")

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
import os
import re # For date checking

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_retries=2,
    api_key=api_key,
)

pdf_agent_prompt_template = """
You are an autonomous PDF Agent. Output a single, valid JSON object. Do NOT use markdown fences or add any commentary.

# Objective
Given raw text from a PDF, determine if it's an INVOICE or POLICY. Extract relevant fields and set flags.

# Core Fields (Common & Invoice/Policy Specific)
* `pdf_type`: (string) "INVOICE" or "POLICY".
* `vendor`: (string or null) For INVOICE.
* `invoice_number`: (string or null) For INVOICE.
* `date`: (string or null, YYYY-MM-DD) For INVOICE (invoice date) or POLICY (effective date).
* `line_items`: (list of objects or null) For INVOICE. Each: `description`, `quantity`, `unit_price`, `line_total` (all numbers for monetary).
* `subtotal`, `tax`, `total`: (number or null) For INVOICE.
* `policy_title`: (string or null) For POLICY.
* `key_terms`: (list of strings or null) For POLICY (e.g., "GDPR", "FDA", "HIPAA", "confidentiality").
* `due_date`: (string or null, YYYY-MM-DD) For INVOICE.

# Flags (Default to "NONE" if not applicable)
* `invoice_flag`: (string) For INVOICE. One of:
    * "HIGH_VALUE_INVOICE": If `total` > 10000.
    * "DUPLICATE_INVOICE": If text explicitly states "DUPLICATE", "COPY OF INVOICE", or similar indication for an already processed invoice number (if provided in context, otherwise rely on text).
    * "INCOMPLETE_INVOICE": If essential fields like `total`, line item details, or `vendor` are clearly missing or marked as 'TBD'/'Pending'.
    * "PAST_DUE_INVOICE": If `due_date` is present and is before the current date (assume current date is {current_date}).
    * "NONE": If no other invoice flags apply.
* `policy_flag`: (string) For POLICY. One of:
    * "COMPLIANCE_RISK": If `key_terms` include "GDPR", "FDA", "HIPAA", "SOX", "PCI-DSS".
    * "LEGAL_REVIEW_REQUIRED": If text explicitly states "requires legal review", "legal approval needed", "consult legal team", or mentions significant liabilities/penalties.
    * "NONE": If no other policy flags apply.

# Instructions
1.  Determine `pdf_type`.
2.  Extract all relevant fields for that type. Dates must be YYYY-MM-DD. Monetary values are numbers.
3.  Evaluate and set `invoice_flag` if INVOICE. Check for keywords indicating duplicate or incompleteness. Compare `due_date` with current date {current_date} for "PAST_DUE_INVOICE".
4.  Evaluate and set `policy_flag` if POLICY.
5.  If a field is not applicable or not found, use `null`.

# Output Format
A single JSON object.
{{
  "pdf_type":       "<INVOICE|POLICY>",
  "vendor":         "<string or null>",
  "invoice_number": "<string or null>",
  "date":           "<YYYY-MM-DD or null>",
  "due_date":       "<YYYY-MM-DD or null>",
  "line_items":     <list or null>,
  "subtotal":       <number or null>,
  "tax":            <number or null>,
  "total":          <number or null>,
  "invoice_flag":   "<HIGH_VALUE_INVOICE|DUPLICATE_INVOICE|INCOMPLETE_INVOICE|PAST_DUE_INVOICE|NONE>",
  "policy_title":   "<string or null>",
  "key_terms":      <list or null>,
  "policy_flag":    "<COMPLIANCE_RISK|LEGAL_REVIEW_REQUIRED|NONE>"
}}

# Examples

## Example 1 (Past Due Invoice)
Current Date: 2025-06-05
Input:
INVOICE
Invoice Number: INV-2025-001
Bill To: Late Payer LLC
Date: 2025-04-01
Due Date: 2025-05-01
Total Due: 500.00
Output:
{{
  "pdf_type": "INVOICE",
  "vendor": "Late Payer LLC",
  "invoice_number": "INV-2025-001",
  "date": "2025-04-01",
  "due_date": "2025-05-01",
  "line_items": null, "subtotal": null, "tax": null,
  "total": 500.00,
  "invoice_flag": "PAST_DUE_INVOICE",
  "policy_title": null, "key_terms": null, "policy_flag": "NONE"
}}

## Example 2 (Policy Requiring Legal Review)
Current Date: 2025-06-05
Input:
INTERNAL MEMORANDUM: PROPOSED VENDOR AGREEMENT POLICY
Effective Date: 2025-07-01
This new policy outlines standard terms for vendor contracts. Section 5 regarding data liability requires legal review before finalization. Key terms: vendor contracts, data liability, GDPR.
Output:
{{
  "pdf_type": "POLICY",
  "vendor": null, "invoice_number": null, "due_date": null,
  "date": "2025-07-01",
  "line_items": null, "subtotal": null, "tax": null, "total": null,
  "invoice_flag": "NONE",
  "policy_title": "INTERNAL MEMORANDUM: PROPOSED VENDOR AGREEMENT POLICY",
  "key_terms": ["vendor contracts", "data liability", "GDPR"],
  "policy_flag": "LEGAL_REVIEW_REQUIRED"
}}

## Example 3 (Incomplete Invoice)
Current Date: 2025-06-05
Input:
DRAFT INVOICE
Invoice Number: DRAFT-002
To: Client X
Date: 2025-06-04
Line Item 1: Service A - Price TBD
Total: PENDING
Output:
{{
  "pdf_type": "INVOICE",
  "vendor": "Client X",
  "invoice_number": "DRAFT-002",
  "date": "2025-06-04",
  "due_date": null,
  "line_items": [{{"description": "Service A", "quantity": null, "unit_price": null, "line_total": null}}],
  "subtotal": null, "tax": null,
  "total": null,
  "invoice_flag": "INCOMPLETE_INVOICE",
  "policy_title": null, "key_terms": null, "policy_flag": "NONE"
}}
---
Current Date: {current_date}
PDF Text Input:
{user_input}

Output:
"""

# It's important to pass the current date to the prompt for "PAST_DUE_INVOICE" logic
# The orchestration script should ideally format this.
# For testing, we can define it here or pass it in the input variables.

def get_current_date_for_prompt():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

prompt = PromptTemplate(
    input_variables=["user_input", "current_date"], # Added current_date
    template=pdf_agent_prompt_template,
)

chain = prompt | llm

def make_chain() -> RunnableSequence:
    return chain

def process_pdf_text(text: str) -> str:
    current_date = get_current_date_for_prompt()
    ai_message = chain.invoke({"user_input": text, "current_date": current_date})
    return ai_message.content.strip()

