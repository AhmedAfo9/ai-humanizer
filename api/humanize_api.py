import os
import re
import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Human Baseline Mimicry Engine", version="8.3")

# إعدادات الـ CORS لضمان استقرار الاتصال مع الواجهة
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

    # هندسة الأوامر (Prompt Engineering) العكسية بناءً على نصوصك البشرية المرجعية
    system_prompt = f"""You are a non-native English academic researcher writing your Master's thesis in linguistics. 
Your goal is to rewrite the given text so it completely matches authentic, unpolished, and direct human academic writing. 

Apply these core human-mimicry rules based on real academic baseline data:
1. Lexical Simplicity (CRITICAL): Use basic, direct verbs and nouns (e.g., 'studied', 'plays a role', 'draw attention', 'emerged', 'make a contribution'). NEVER use AI-typical academic jargon like 'delve', 'underscores', 'paramount', 'elucidates', 'tapestry', 'pivotal', 'fosters', 'demise', 'unfurls', or 'cascade'.
2. Grammatical Authenticity & Unpolished Phrasing: Do not over-polish the grammar. Use straightforward, sometimes slightly rigid phrasing that focuses purely on meaning (e.g., 'is firstly emerged', 'have a very important role'). Avoid complex, flowery, or passive-aggressive AI structures.
3. Structural Repetition: It is completely acceptable and encouraged to repeat transition phrases or structural framing in consecutive sentences (e.g., 'It is hoped that...', 'This study is also hoped to...'). Do not use a thesaurus to artificially vary every single word. Humans repeat structures naturally.
4. Sentence Flow: Write plain declarative sentences. Avoid starting sentences with complex dependent clauses or participial phrases (e.g., 'Reeling from his father's death...'). Just state the subject and verb directly.
5. Absolute Guardrails: Do not modify names, years, locations, or citations (e.g., '(Abraham, 2016)').

Your output must be the raw rewritten text only, preserving original line breaks and citation formats exactly. Do not include any meta-commentary, introductory text, or markdown code blocks."""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Academic Baseline Mimicry Suite"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.75, # خفضنا الحرارة قليلاً لمنع الموديل من "الإبداع" وإجباره على التبسيط الحرفي
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
    
    # تنظيف الفراغات حول علامات الترقيم
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
    return {"status": "healthy", "engine": "Human Baseline Engine v8.3"}
