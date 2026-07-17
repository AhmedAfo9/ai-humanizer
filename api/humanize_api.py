import os
import re
import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Ultimate Academic Humanizer API Powered by OpenRouter", version="7.8")

# --- إعدادات الـ CORS المستقرة لمنع مشاكل الاتصال مع كلاودفلير الواجهة ---
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

# --- مساعدات خفيفة لحساب الإحصائيات بدون استهلاك الذاكرة (RAM) ---
def count_words(text: str) -> int:
    return len(text.split())

def count_sentences(text: str) -> int:
    # تقسيم ذكي وبسيط للجمل بناءً على علامات الترقيم لضمان خفة السيرفر
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

# --- المحرك الذكي للاتصال بـ OpenRouter وترجمة خيارات المستخدم ---
def call_openrouter_engine(text: str, p_syn: float, p_trans: float, use_passive: bool) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY is not set in Render Environment Variables."

    # 1. هندسة التوجيه الخاص بقوة المفردات بناءً على شريط التحكم
    if p_syn <= 0.3:
        vocab_instruction = "Keep the original vocabulary mostly intact. Only rephrase phrases that sound clunky, ensuring basic academic clarity without changing technical terms."
    elif p_syn <= 0.6:
        vocab_instruction = "Apply moderate vocabulary enhancement. Substitute simple or repetitive words with advanced, context-aware academic synonyms. Do NOT use archaic or obsolete words like 'rex'."
    else:
        vocab_instruction = "Extensively rewrite and elevate the text using a highly sophisticated academic lexicon and professional phrasing, completely reshaping the sentence style while holding the core meaning."

    # 2. هندسة التوجيه الخاص بروابط التدفق بناءً على شريط التحكم
    if p_trans <= 0.3:
        transition_instruction = "Allow sentences to flow naturally with minimal explicit transition words; rely on syntax for organic progression."
    elif p_trans <= 0.6:
        transition_instruction = "Integrate standard academic transitional phrases (e.g., Moreover, Consequently, Furthermore, Therefore, Notably) naturally at clause and sentence boundaries where logically appropriate."
    else:
        transition_instruction = "Maximize the usage of diverse and advanced academic discourse markers to explicitly and heavily connect every logical turn and argument."

    # 3. هندسة التوجيه الخاص بسويتش الـ Passive Voice للتحويل الهيكلي الحقيقي
    if use_passive:
        passive_instruction = "Actively convert appropriate active voice clauses into elegant, formal academic passive voice structures. This structural inversion is critical to break AI structural patterns completely."
    else:
        passive_instruction = "Maintain a natural academic balance between active and passive voice without forced inversion."

    # 4. بناء الـ System Prompt الاحترافي والأكاديمي المنيع
    system_prompt = f"""You are an elite academic editor and native English proofreader specializing in advanced computational linguistics. 
Your objective is to humanize the user's text to bypass AI detectors completely while elevating its scholarly quality.

You must strictly fulfill these dynamic parameters selected via the interface:
- Vocabulary Alteration Level: {vocab_instruction}
- Flow & Discursive Markers Level: {transition_instruction}
- Syntactic Restructuring (Passive Voice): {passive_instruction}

CRITICAL LINGUISTIC AND STRUCTURAL COMMANDS:
1. NEVER alter, remove, space out, or find synonyms for Proper Nouns, historical figures, names of characters (e.g., 'Hamlet', 'Claudius', 'Shakespeare'), titles of literary works, specific countries/locations, or technical jargon.
2. Flawless Typographical Execution: Do NOT inject weird spaces before or after punctuation or possessive clitics. Write "Shakespeare's" (NOT "Shakespeare 's"), "To be, or not to be," (NOT "To be , or non to be , ").
3. Completely preserve the paragraph structure and line breaks of the input text.
4. Output ONLY the finalized humanized text. Do NOT wrap it in markdown blockquotes, do NOT add introduction words, notes, greetings, or backticks (no ```text or ```html)."""

    # 5. إعداد الطلب وتجهيز الموديل الفائق Gemini 2.5 Flash
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Academic Humanizer Suite"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash",  # نموذج ذكي جداً لغوياً، سريع، وشبه مجاني في التكلفة
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.65  # توازن مثالي لمنع الشطحات الدلالية غير المنضبطة
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
    # حساب الإحصائيات قبل التعديل
    orig_words = count_words(request.text)
    orig_sentences = count_sentences(request.text)
    
    # معالجة النص عبر خادم العقل اللغوي الذكي المربوط بالـ API
    result_text = call_openrouter_engine(
        text=request.text,
        p_syn=request.p_syn,
        p_trans=request.p_trans,
        use_passive=request.use_passive
    )
    
    # تنظيف سريع وميكانيكي أخير للتأكد من خلو علامات الترقيم من أي فراغات تافهة
    result_text = re.sub(r"\s+([.,;:!?])", r"\1", result_text)
    result_text = re.sub(r"``\s*(.+?)\s*''", r'"\1"', result_text)
    
    # حساب الإحصائيات بعد التعديل لترجع بانتظام لواجهة الهاتف
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
    return {"status": "healthy", "engine": "OpenRouter Linguistic Engine v7.8"}
