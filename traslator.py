import os
import re

_client = None

_ROMAN_URDU_MARKERS = {
    "hai",
    "hain",
    "ho",
    "kya",
    "ka",
    "ki",
    "ke",
    "ap",
    "aap",
    "mujhe",
    "mujhy",
    "chahiye",
    "chahiyeh",
    "do",
    "joote",
    "sasti",
    "mehnga",
    "kahan",
    "kitna",
    "wala",
    "wali",
}


def get_client():
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            _client = False
        else:
            from groq import Groq

            _client = Groq(api_key=api_key)

    return None if _client is False else _client


def _looks_like_english(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", (text or "").lower())
    if not words:
        return False

    marker_hits = sum(1 for word in words if word in _ROMAN_URDU_MARKERS)
    if marker_hits:
        return False

    return len(words) >= 3 or any(word in {"do", "you", "have", "need", "want", "show", "find"} for word in words)


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
    cleaned_input = (text or "").strip()
    if not cleaned_input:
        return cleaned_input

    if _looks_like_english(cleaned_input):
        return cleaned_input

    prompt = f"Roman Urdu query: {cleaned_input}"

    client = get_client()
    if not client:
        return cleaned_input

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

    translated = response.choices[0].message.content if response.choices else cleaned_input
    return _normalize_translation_output(translated)