import os
import re
import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Elite Prose Editing Engine", version="8.0")

# --- إعدادات الـ CORS المستقرة لمنع مشاكل الاتصال ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str
    p_syn: Optional[float] = 0.2
    p_trans: Optional[float] = 0.2
    preserve_linebreaks: Optional[bool] = True
    use_passive: Optional[bool] = False

def count_words(text: str) -> int:
    return len(text.split())

def count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

def call_openrouter_engine(text: str, p_syn: float, p_trans: float, use_passive: bool) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY is not set."

    # 1. صياغة ديناميكية طبيعية للمفردات بعيداً عن المصطلحات المحنطة
    if p_syn <= 0.3:
        vocab_instruction = "Make minimal adjustments, only smoothing out stiff phrases while keeping the core vocabulary intact."
    elif p_syn <= 0.6:
        vocab_instruction = "Moderate editing. Replace repetitive or textbook words with natural, vivid, and precise conversational alternatives."
    else:
        vocab_instruction = "Thoroughly rewrite the text. Use highly diverse, rich human expressions and crisp idioms to completely refresh the vocabulary."

    # 2. إدارة روابط الجمل بشكل طبيعي وسلس جداً
    if p_trans <= 0.3:
        transition_instruction = "Do not use formal transition words. Let sentences connect through pure logic and natural narrative progression."
    elif p_trans <= 0.6:
        transition_instruction = "Use light, organic conversational connectors (like so, yet, but, instead, because of this) only when necessary to bridge thoughts."
    else:
        transition_instruction = "Ensure seamless prose flow using a rich variety of structural links, keeping it completely natural and non-repetitive."

    # 3. دمج المبني للمجهول بذكاء بشري متمكن
    if use_passive:
        passive_instruction = "Blend elegant passive voice constructions into the paragraphs to shift focus to the actions and objects, effectively breaking up repetitive sentence cadences."
    else:
        passive_instruction = "Favor active, direct verbs, keeping passive phrasing only where it feels entirely organic to standard English."

    # 4. الـ System Prompt الشبح (لا يحتوي على أي إشارة للذكاء الاصطناعي أو الكواشف)
    system_prompt = f"""You are a master copyeditor and a seasoned storyteller rewriting a rough draft for publication in a premium magazine. Your goal is to make the prose sound exceptionally vivid, natural, and engaging, capturing the authentic rhythm of native human speech.

Apply these tailored editing preferences based on the configuration:
- Vocabulary Modulation: {vocab_instruction}
- Structural Flow: {transition_instruction}
- Voice Configuration: {passive_instruction}

CORE RULES FOR NATURAL HUMAN PROSE:
1. Sentence Length Burstiness (Crucial): Heavily vary your sentence structures. Follow a long, descriptive sentence with a very short, sharp, punchy one (4-7 words). Uniform sentence length is completely prohibited.
2. Dynamic Openings: Never start consecutive sentences with the same subject or pronoun. Mix your sentence openings (e.g., start with a dependent clause, a prepositional phrase, or a direct action).
3. Absolute Clarity: Write with crisp, clear, and direct language. Avoid overly ornamental, pretentious, or robotic jargon that sounds like a machine trying too hard.
4. Rigid Guardrails: NEVER change, alter, or add spaces to proper nouns, character names ('Hamlet', 'Shakespeare', 'Claudius'), geographic locations ('Denmark'), or citation placeholders like [[REF_1]]. 
5. Maintain original paragraphs and line breaks perfectly.
6. Output ONLY the raw, polished text. No introductions, no notes, and no markdown formatting wrappers."""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Academic Humanizer Suite"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.85,  # رفع الحرارة لكسر الأنماط الرياضية المتوقعة وتدمير بصمة الـ AI
        "max_tokens": 4000
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=45)
        if response.status_code == 200:
            result_json = response.json()
            return result_json['choices'][0]['message']['content'].strip()
        else:
            return f"OpenRouter API Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"Backend Connection Error: {str(e)}"


@app.post("/humanize")
async def humanize_endpoint(request: HumanizeRequest):
    orig_words = count_words(request.text)
    orig_sentences = count_sentences(request.text)
    
    result_text = call_openrouter_engine(
        text=request.text,
        p_syn=request.p_syn,
        p_trans=request.p_trans,
        use_passive=request.use_passive
    )
    
    # تنظيف ميكانيكي للفراغات قبل الترقيم لضمان مظهر بشري ممتاز
    result_text = re.sub(r"\s+([.,;:!?])", r"\1", result_text)
    
    new_words = count_words(result_text)
    new_sentences = count_sentences(result_text)
    
    return {
        "humanized_text": result_text,
        "orig_word_count": orig_words,
        "orig_sentence_count": orig_sentences,
        "new_word_count": new_words,
        "new_sentence_count": new_sentences,
        "words_added": max(0, new_words - orig_words),
        "sentences_added": max(0, new_sentences - orig_sentences)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "Stealth Editing Engine v8.0"}
