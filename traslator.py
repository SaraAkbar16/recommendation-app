import os
import re
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _normalize_translation_output(text: str) -> str:
    cleaned = (text or "").strip()

    # Remove common wrapper phrases that hurt search quality.
    patterns = [
        r'^the roman urdu translates to:\s*["\']?(.*?)["\']?$',
        r'^translation:\s*["\']?(.*?)["\']?$',
        r'^english:\s*["\']?(.*?)["\']?$',
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

    return cleaned.strip('"\' ')

def roman_urdu_to_english(text: str) -> str:
    prompt = f"Roman Urdu query: {text}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Roman Urdu to English translator for e-commerce search. "
                    "Return only the translated query text, not explanations, not labels, not quotes. "
                    "Keep it short and literal. Preserve product intent, product type, color, size, and brand words. "
                    "If text is already English, return it unchanged."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    translated = response.choices[0].message.content if response.choices else text
    return _normalize_translation_output(translated)