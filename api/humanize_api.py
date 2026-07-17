import os
import re
import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Anti-AI Academic Humanizer Engine", version="7.9")

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
        return "Error: OPENROUTER_API_KEY is not set in Render Environment Variables."

    # 1. إعادة هندسة شريط المفردات للتركيز على إعادة الصياغة البشرية المباشرة وليس الفخامة الزائفة
    if p_syn <= 0.3:
        vocab_instruction = "Keep the phrasing close to the original draft, only clarifying sentences that feel stiff or robotic."
    elif p_syn <= 0.6:
        vocab_instruction = "Rephrase sentences using natural, clear, and direct alternative words. Use clean, clear academic verbs."
    else:
        vocab_instruction = "Completely break down and rebuild the sentences using highly diverse, natural human phrasing. Write with dynamic and crisp idioms."

    # 2. إعادة هندسة شريط الروابط وحظر أدوات الربط الآلية المتكررة
    if p_trans <= 0.3:
        transition_instruction = "Do NOT use explicit transition words. Let the logic flow naturally through punctuation and clean sentence progression."
    elif p_trans <= 0.6:
        transition_instruction = "Use simple, organic conversational transitions (e.g., So, Thus, As a result, Meanwhile, Because of this) only where absolutely necessary."
    else:
        transition_instruction = "Connect logical arguments smoothly using varied connectors, but rigidly avoid mechanical repetition."

    # 3. هندسة تحويل البنية النحوية للمبني للمجهول بشكل غير مكشوف
    if use_passive:
        passive_instruction = "Convert appropriate active clauses into crisp, standard academic passive voice structures. Invert sentence subjects to radically shatter the predictable AI syntactic structure."
    else:
        passive_instruction = "Maintain a natural and unforced balance between active and passive verbs."

    # 4. بناء الـ System Prompt الجديد (صائد ومفتت رادارات الـ AI الكاشفة)
    system_prompt = f"""You are a human academic writer rewriting a draft to sound completely natural, organic, and direct. 
Your primary goal is to strip away all computational footprints and bypass AI detectors like Turnitin and GPTZero.

AI detectors flag text that uses "thesaurus-stuffing" (overly ornamental, fancy words) or rigid, robotic transitions. Real humans write clearly, utilizing active verbs, direct expressions, and immense sentence length variety.

Adhere to these interface parameters selected by the user:
- Phrasing Variance Shift: {vocab_instruction}
- Discourse Connection Type: {transition_instruction}
- Voice Inversion (Passive Mode): {passive_instruction}

STRICT ANTI-AI LINGUISTIC RULES (CRITICAL):
1. MANDATORY BLACKLIST: Do NOT, under any circumstance, use any of these typical AI signature words: delve, tapestry, testament, pivotal, renowned, paramount, furthermore, moreover, consequently, subsequently, unfolds, chronicling, demise, weds, ascends to the throne, specter, perpetrated, exact vengeance, grapples, eloquent, internal turmoil, philosophical contemplations, plot progresses, retaliatory, fatalities, principal characters, perfidy, fidelity, ramifications, endures, preeminent, staged globally, ultimate, multifaceted, underscore. They are immediate AI triggers.
2. Use clear, concise human terminology. Instead of "demise" use "death"; instead of "narrative unfolds" use "story is set in" or "takes place"; instead of "exact vengeance" use "get revenge".
3. Structural Burstiness: Mix short, punchy sentences (4 to 7 words) with medium and longer sentences. Avoid having sentences of uniform length. This variation is the #1 human signature.
4. NEVER alter, touch, or add spaces to citation placeholders like [[REF_1]], [[REF_2]]. 
5. Keep names and locations untouched ('Hamlet', 'Shakespeare', 'Denmark', 'Claudius').
6. Preserve original line breaks and paragraphs exactly.
7. Output ONLY the raw humanized text. Do NOT wrap it in markdown code blocks, and do NOT include any introductory or conversational notes."""

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
        "temperature": 0.78,  # رفع حرارة الإبداع قليلاً لزيادة الحيرة اللغوية (Perplexity) ومنع الأنماط المتكررة
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
    
    # تنظيف ميكانيكي أخير للفراغات العشوائية حول علامات الترقيم
    result_text = re.sub(r"\s+([.,;:!?])", r"\1", result_text)
    result_text = re.sub(r"``\s*(.+?)\s*''", r'"\1"', result_text)
    
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
    return {"status": "healthy", "engine": "Anti-AI Humanizer Engine v7.9"}
