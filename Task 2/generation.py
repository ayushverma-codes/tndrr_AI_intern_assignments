import re
import time
from groq import Groq, RateLimitError, APIError
from config import api_key, GROQ_MODEL, ABSTAIN
def extract_numerical_claims(text):
    """
    Extract explicit numerical claims from generated text.

    Handles:
    - Monetary amounts: $50,000
    - Percentages: 25%
    - Other numerical values: 3.5, 2024, 1000

    This is a conservative lexical check,
    not numerical reasoning.
    """

    pattern = r"(?<!\w)\$?\d[\d,]*(?:\.\d+)?%?"

    return re.findall(pattern, text)

def normalize_number(value):
    return value.replace(",", "").strip()

def verify_generated_numbers(answer, evidence):
    """
    Verify whether numbers in the generated answer
    appear in the evidence supplied to the LLM.
    """

    answer_numbers = extract_numerical_claims(answer)

    evidence_text = " ".join(
        item["text"]
        for item in evidence
    )

    evidence_numbers = {
        normalize_number(number)
        for number in extract_numerical_claims(
            evidence_text
        )
    }

    unsupported = [
        number
        for number in answer_numbers
        if normalize_number(number)
        not in evidence_numbers
    ]

    return {
        "passed": len(unsupported) == 0,
        "answer_numbers": answer_numbers,
        "unsupported_numbers": unsupported
    }
def normalize_citations(answer):
    return re.sub(r'【E(\d+)】', r'[E\1]', answer)

def validate_answer(answer, evidence):
    answer = normalize_citations(answer)
    ids = re.findall(r'\[E(\d+)\]', answer)
    abstained = answer.rstrip('.').lower() == ABSTAIN.rstrip('.').lower()
    valid = bool(ids) and all(1 <= int(n) <= len(evidence) for n in ids)
    numbers = verify_generated_numbers(answer, evidence)
    reason = 'llm_abstention' if abstained else None
    if not abstained and not valid:
        reason = 'invalid_or_missing_citation'
    elif not abstained and not numbers['passed']:
        reason = 'unsupported_numerical_claim'
    return dict(answer=ABSTAIN if reason else answer, abstained=bool(reason), abstention_reason=reason,
                citation_format_pass=abstained or valid, number_verification=numbers,
                cited_passages=[] if reason else [item for i,item in enumerate(evidence,1) if str(i) in ids])

class Generator:
    def __init__(self):
        self.client = Groq(api_key=api_key(), timeout=30, max_retries=0)
        self.last_call = 0.0

    def generate(self, question, evidence):
        context = '\n\n'.join(f"[E{i}] PDF page {e['page']}\n{e['text']}" for i,e in enumerate(evidence,1))
        messages = [dict(role='system', content=(
            'You answer questions using only the supplied NIST AI RMF source passages. '
            'Answer directly in one to three sentences. Include every important part of the answer. '
            'Cite supporting passages with [E1], [E2], etc. Never invent facts or citations. '
            'If evidence is insufficient, reply exactly: ' + ABSTAIN)),
            dict(role='user', content=f'Question:\n{question}\n\nEvidence:\n{context}\n\nProvide a concise, fully supported answer.')]
        time.sleep(max(0, 6 - (time.monotonic() - self.last_call)))
        for attempt in range(3):
            try:
                self.last_call = time.monotonic()
                response = self.client.chat.completions.create(model=GROQ_MODEL, messages=messages,
                    reasoning_effort='low', temperature=0, max_completion_tokens=350)
                break
            except RateLimitError:
                if attempt == 2:
                    raise RuntimeError('Groq rate limit reached. Wait before trying again.') from None
                time.sleep(30)
            except APIError:
                raise RuntimeError('Groq request failed. Check connectivity, credentials, and service availability.') from None
        raw = (response.choices[0].message.content or '').strip()
        result = validate_answer(raw, evidence)
        result.update(raw_answer=raw, llm_called=True, usage=response.usage.model_dump(), finish_reason=response.choices[0].finish_reason)
        return result
