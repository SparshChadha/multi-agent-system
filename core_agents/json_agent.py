# json_agent.py
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("GEMINI_KEY")

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_retries=2,
    api_key=api_key,
)

json_agent_prompt_template = """
You are an autonomous JSON Agent. **Immediately** output a single, valid JSON object. Do NOT use markdown fences or add any commentary.

# Objective
Given raw text of a JSON webhook, parse it, validate its schema, and classify its status.

# Schema Definition (Required Fields for "VALID" status):
* `event`: (string) The type of event (e.g., "order_placed", "user_updated", "payment_failed").
* `order_id`: (string, conditionally required if event relates to an order)
* `amount`: (number, conditionally required if event involves a transaction)
* `currency`: (string, conditionally required if event involves a transaction)
* `customer_id`: (string, conditionally required if event relates to a customer)
* `payload_version`: (string, e.g., "v2", "v3")

# Validation Statuses:
1.  `INVALID_JSON`: Input string is not parsable as JSON.
2.  `INVALID_SCHEMA`: Parsable JSON, but misses required fields OR has type mismatches for core fields like `event` or `payload_version`.
3.  `DEPRECATED_SCHEMA`: `payload_version` is present and indicates an old version (e.g., "v1" when current is "v2" or "v3"). The current schema versions are "v2" and "v3".
4.  `UNKNOWN_EVENT`: `event` field contains a value not in the known list: ["order_placed", "order_shipped", "order_cancelled", "user_created", "user_updated", "payment_succeeded", "payment_failed", "item_added_to_cart"].
5.  `VALID`: JSON is parsable, conforms to the schema (all absolutely required fields like `event` and `payload_version` are present and correctly typed), `payload_version` is current, and `event` is known.

# Instructions
1.  Attempt to parse `user_input` as JSON. If fails, output `INVALID_JSON` status.
2.  If parsable, check for `event` and `payload_version` fields.
    * If `payload_version` is "v1", set status to `DEPRECATED_SCHEMA`.
    * If `event` is not in the known list, set status to `UNKNOWN_EVENT`.
    * If `event` or `payload_version` are missing or have wrong types, set status to `INVALID_SCHEMA`. Include specific errors.
3.  If `event` and `payload_version` are valid and current, and other core fields are present and typed correctly, status is `VALID`.
4.  For `INVALID_SCHEMA`, list missing fields or type mismatches in `"errors"`.
5.  For `VALID`, `DEPRECATED_SCHEMA`, `UNKNOWN_EVENT`, the `"data"` field should contain the parsed JSON. For `INVALID_JSON` or critical `INVALID_SCHEMA` (e.g. missing 'event'), data can be null or the partial parse.

# Output Format
A single JSON object.
{{
  "validation_status": "<VALID|INVALID_JSON|INVALID_SCHEMA|DEPRECATED_SCHEMA|UNKNOWN_EVENT>",
  "errors":            <["<string>", ...] or null>,
  "warnings":          <["<string>", ...] or null>, // For non-critical issues like extra fields
  "data":              <parsed JSON object or null>
}}

# Examples

## Example 1 (Valid JSON - Current Schema)
Input:
{{
  "event": "order_placed",
  "payload_version": "v2",
  "order_id": "ORD-5678",
  "amount": 1250.50,
  "currency": "USD",
  "customer_id": "CUST-9012"
}}
Output:
{{
  "validation_status": "VALID",
  "errors": [],
  "warnings": [],
  "data": {{
    "event": "order_placed",
    "payload_version": "v2",
    "order_id": "ORD-5678",
    "amount": 1250.5,
    "currency": "USD",
    "customer_id": "CUST-9012"
  }}
}}

## Example 2 (Deprecated Schema)
Input:
{{
  "event": "user_updated",
  "payload_version": "v1",
  "customer_id": "CUST-001",
  "details": {{"email": "new@example.com"}}
}}
Output:
{{
  "validation_status": "DEPRECATED_SCHEMA",
  "errors": [],
  "warnings": ["Payload version v1 is deprecated. Please upgrade to v2 or v3."],
  "data": {{
    "event": "user_updated",
    "payload_version": "v1",
    "customer_id": "CUST-001",
    "details": {{"email": "new@example.com"}}
  }}
}}

## Example 3 (Unknown Event)
Input:
{{
  "event": "user_logged_out",
  "payload_version": "v3",
  "customer_id": "CUST-002"
}}
Output:
{{
  "validation_status": "UNKNOWN_EVENT",
  "errors": ["Event type 'user_logged_out' is not recognized."],
  "warnings": [],
  "data": {{
    "event": "user_logged_out",
    "payload_version": "v3",
    "customer_id": "CUST-002"
  }}
}}

## Example 4 (Invalid Schema - Missing Core Field)
Input:
{{
  "payload_version": "v2",
  "order_id": "ORD-FAIL-001"
}}
Output:
{{
  "validation_status": "INVALID_SCHEMA",
  "errors": ["Missing required field: event"],
  "warnings": [],
  "data": {{
    "payload_version": "v2",
    "order_id": "ORD-FAIL-001"
  }}
}}

---
JSON Input:
{user_input}

Output:
"""

prompt = PromptTemplate(input_variables=["user_input"], template=json_agent_prompt_template)

chain = prompt | llm

def make_chain() -> RunnableSequence:
    return chain

def validate_json_payload(text: str) -> str:
    ai_message = chain.invoke({"user_input": text})
    return ai_message.content.strip()

