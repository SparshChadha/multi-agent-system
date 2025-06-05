from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("GEMINI_KEY")

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
import os

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_retries=2,
    api_key=api_key,
)

email_agent_prompt_template = """
You are an autonomous Email Agent. **Immediately** output a single, valid JSON object matching the specified schema. Do NOT use markdown fences or add any commentary.

# Objective
Given an email's content, extract specific fields, classify its properties, and determine a primary action.

# Fields to Extract & Classify:
1.  `sender`: (string) Email address of the sender (e.g., "user@example.com"). If not found, use "unknown@sender.com".
2.  `urgency`: (string) One of "high", "medium", "low".
    * "high": Keywords like "URGENT", "ASAP", "immediately", "critical", "now", or phrases indicating severe impact/tight deadlines.
    * "low": Keywords like "no rush", "whenever possible", "at your convenience".
    * "medium": Default if no strong cues for high/low, or standard requests.
3.  `issue_summary`: (string) A concise, one-sentence summary of the email's main point or request.
4.  `tone`: (string) One of "escalation", "polite", "threatening".
    * "escalation": Angry, demanding language ("unacceptable", "I demand", "fix this now"), excessive ALL CAPS/exclamation marks, strong impatience.
    * "threatening": Explicit negative consequences ("I will sue", "legal action", "report you").
    * "polite": Calm, courteous, standard business/personal correspondence. Prioritize severe tones if mixed.
5.  `department`: (string) Infer the most relevant department. One of "billing", "support", "sales", "security", "general_inquiry".
    * "billing": Inquiries about invoices, payments, charges, refunds.
    * "support": Technical issues, product help, complaints about service/product.
    * "sales": Product inquiries, pricing, new orders, partnership requests.
    * "security": Reports of phishing, suspicious activity, account compromise.
    * "general_inquiry": If none of the above fit clearly.
6.  `spam_flag`: (string) One of "CLEAR", "SUSPICIOUS", "PHISHING_RISK".
    * "PHISHING_RISK": Mentions of account compromise, urgent requests to click suspicious links, requests for login details, poor grammar from an official-looking source, mismatched sender domain.
    * "SUSPICIOUS": Unsolicited marketing, unusual requests, vague content from unknown sender.
    * "CLEAR": Appears to be legitimate communication.
7.  `action`: (string) The primary suggested action based on analysis. One of "ESCALATE", "ACKNOWLEDGE", "ROUTINE".
    * If `tone` is "escalation" OR "threatening" OR `urgency` is "high" OR `spam_flag` is "PHISHING_RISK" → "ESCALATE".
    * If `spam_flag` is "CLEAR" AND `urgency` is "low" AND `tone` is "polite" AND department is "general_inquiry" or "sales" (for simple info requests) → "ACKNOWLEDGE".
    * Otherwise (including "SUSPICIOUS" spam that isn't high phishing risk) → "ROUTINE".

# Output Format
A single JSON object. Ensure all specified keys are present.
{{
  "sender":        "<email_address>",
  "urgency":       "<high|medium|low>",
  "issue_summary": "<short text summary>",
  "tone":          "<escalation|polite|threatening>",
  "department":    "<billing|support|sales|security|general_inquiry>",
  "spam_flag":     "<CLEAR|SUSPICIOUS|PHISHING_RISK>",
  "action":        "<ESCALATE|ACKNOWLEDGE|ROUTINE>"
}}

# Examples

## Example 1 (Phishing Email)
Input:
From: secuirty@paypal-servicing-account.com
Subject: Urgent: Your account is suspended!
Body: Dear customer, we detect unusual login from your PqyPal account. Please login immediately via this link to restore access: http://fakelink.com/paypal/login

Output:
{{
  "sender": "secuirty@paypal-servicing-account.com",
  "urgency": "high",
  "issue_summary": "Claims PayPal account is suspended and asks to login via a suspicious link.",
  "tone": "escalation",
  "department": "security",
  "spam_flag": "PHISHING_RISK",
  "action": "ESCALATE"
}}

## Example 2 (Billing Inquiry)
Input:
From: jane.doe@email.com
Subject: Question about Invoice #123
Body: Hi, I was looking at my last bill (Invoice #123) and I think there's a mistake in the service charge for Project X. Can someone from billing take a look? Thanks!

Output:
{{
  "sender": "jane.doe@email.com",
  "urgency": "medium",
  "issue_summary": "User questions a service charge on Invoice #123 and requests billing review.",
  "tone": "polite",
  "department": "billing",
  "spam_flag": "CLEAR",
  "action": "ROUTINE"
}}

## Example 3 (Simple Acknowledgment)
Input:
From: prospect@company.com
Subject: Info on your services
Body: Hello, could you please send me a brochure of your enterprise solutions? No particular rush. Thank you.

Output:
{{
  "sender": "prospect@company.com",
  "urgency": "low",
  "issue_summary": "Requests a brochure of enterprise solutions.",
  "tone": "polite",
  "department": "sales",
  "spam_flag": "CLEAR",
  "action": "ACKNOWLEDGE"
}}
---
Email Input:
{user_input}

Output:
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    template=email_agent_prompt_template,
)

chain = prompt | llm

def make_chain() -> RunnableSequence:
    return chain

def process_email(text: str) -> str:
    ai_message = chain.invoke({"user_input": text})
    return ai_message.content.strip()
