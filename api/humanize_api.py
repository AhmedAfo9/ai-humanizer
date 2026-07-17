import os
import re
import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="C1 Academic Advanced Editing Engine", version="8.2")

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

    # 1. ضبط معايير اختيار الكلمات بناءً على مستوى C1 الطبيعي والنظيف
    if p_syn <= 0.3:
        vocab_instruction = "Keep the vocabulary simple, clean, and close to the draft, only fixing awkward phrasing."
    elif p_syn <= 0.6:
        vocab_instruction = "Use natural, common academic verbs. Replace complex textbook metaphors with direct, accurate vocabulary."
    else:
        vocab_instruction = "Fully rewrite using clear, diverse, and robust C1-level expressions, ensuring total clarity without puffery."

    # 2. صياغة روابط منطقية غير متكلفة
    if p_trans <= 0.3:
        transition_instruction = "Do not use robotic transition words. Connect ideas purely through punctuation and sentence order."
    elif p_trans <= 0.6:
        transition_instruction = "Incorporate simple, natural links (like so, yet, because of this, instead) only where logically required."
    else:
        transition_instruction = "Maintain dynamic logical flow using varied, clean structural connectors, completely avoiding mechanical repetition."

    # 3. إدراج الصيغ الهيكلية للمبني للمجهول بذكاء الطالب المتمكن
    if use_passive:
        passive_instruction = "Integrate precise passive voice constructions into appropriate clauses. This structural shift is necessary to break up predictable active-voice patterns."
    else:
        passive_instruction = "Use standard active and passive structures in a balanced, unforced human proportion."

    # 4. الـ System Prompt الجديد كلياً: شخصية الباحث المتقدم C1 (صائد ومفتت البصمات الآلية)
    system_prompt = f"""You are a brilliant non-native academic researcher writing at an advanced C1 English proficiency level. Your prose is structurally flawless, grammatically perfect, and highly coherent, but it is entirely free from pretentious, ornamental, or dramatic AI-jargon. You write with clinical clarity, using straightforward and objective language.

Apply these mechanical preferences based on the configuration:
- Vocabulary Scale: {vocab_instruction}
- Logical Linkage: {transition_instruction}
- Grammatical Voice: {passive_instruction}

STRICT DIRECTIVES TO DESTROY THE AI FOOTPRINT:
1. THE PRECISE BLACKLIST: Do NOT use overly dramatic, narrative, or cliché words that machines rely on to sound smart. Absolutely forbid: unfurls, plunges us into, heart of, web of sorrow, chilling truth, cascade, demise, weds, ascends to the throne, specter, perpetrated, inner turmoil, deeply saddened, profound sorrow, plot progresses, revolutionary, tapestry, testament, delve, pivotal, renowned, paramount, furthermore, moreover, consequently, subsequently, underscores.
2. The C1 Clarity Rule: Write plainly and directly. Instead of "unfurls in Denmark," write "takes place in Denmark." Instead of "caught in a web of sorrow," write "is deeply grieved by." Avoid flowery adjectives; real scholars focus on clear facts and data.
3. Radical Sentence Length Burstiness: Ensure a highly unpredictable rhythm. Mix very long complex sentences (25-30 words) with very short, punchy, declarative sentences (4-6 words). Uniform or predictable sentence length is the primary trigger for AI detection detectors.
4. Rigid Guardrails: NEVER touch, alter, or introduce spaces to proper nouns, character names ('Hamlet', 'Shakespeare'), geographic locations, or citation numbers/tags.
5. Retain original line breaks and paragraphs perfectly.
6. Output ONLY the clean rewritten text, with no markdown code blocks, no introductions, and no conversational notes."""

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
        "temperature": 0.85,  # مستوى مثالي لرفع الحيرة الإحصائية (Perplexity) دون خسارة المعنى
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
    
    # تنظيف الفراغات حول علامات الترقيم لضمان مظهر احترافي
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
    return {"status": "healthy", "engine": "C1 Advanced Stealth Engine v8.2"}
