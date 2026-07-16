import os
import re
import random
import nltk
import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from nltk.corpus import wordnet

# --- تحميل الحزم اللغوية تلقائياً عند إقلاع السيرفر ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# تحميل قاعدة بيانات المترادفات فقط وبشكل هادئ ومباشر
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

app = FastAPI(title="Advanced Academic AI Text Humanizer API", version="2.0")

# --- إعدادات الـ CORS الأكثر موثوقية والأسهل للنشر مستقبلاً ---
# نستخدم "*" وابطال الـ Credentials لضمان عدم حظر المتصفحات للطلبات مطلقاً ولتسهيل تغيير الدومينات
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- هيكل البيانات المستقبلة ---
class HumanizeRequest(BaseModel):
    text: str
    p_syn: Optional[float] = 0.2
    p_trans: Optional[float] = 0.2
    preserve_linebreaks: Optional[bool] = True

# --- قائمة الروابط الأكاديمية الذكية (منعاً للتكرار) ---
ACADEMIC_TRANSITIONS = [
    "Furthermore", "Moreover", "In addition", "Consequently", 
    "Therefore", "On the other hand", "Notably", "In this context"
]

# --- دالة جلب المترادفات الذكية سياقياً ---
def get_semantic_synonym(word, pos_tag):
    wn_tag = None
    if pos_tag.startswith("NN"): wn_tag = wordnet.NOUN
    elif pos_tag.startswith("VB"): wn_tag = wordnet.VERB
    elif pos_tag.startswith("JJ"): wn_tag = wordnet.ADJ
    elif pos_tag.startswith("RB"): wn_tag = wordnet.ADV
    
    if not wn_tag:
        return word

    synonyms = []
    try:
        for syn in wordnet.synsets(word, pos=wn_tag):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if word.istitle(): name = name.title()
                if name.lower() != word.lower() and name not in synonyms:
                    synonyms.append(name)
    except Exception:
        return word
                
    return random.choice(synonyms) if synonyms else word

# --- الدالة الجوهرية للأنسنة ---
def advanced_humanizer(text, p_syn, p_trans):
    citation_pattern = r'(\([A-Za-z\s\.\,]+,\s+\d{4}\)|\[\d+\])'
    citations = re.findall(citation_pattern, text)
    
    for i, citation in enumerate(citations):
        text = text.replace(citation, f" __CITATION_{i}__ ")

    doc = nlp(text)
    processed_words = []
    
    protected_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

    for token in doc:
        if "__CITATION_" in token.text or token.text.lower() in protected_entities or token.is_punct or token.is_digit:
            processed_words.append(token.text)
            continue
            
        if random.random() < p_syn:
            synonym = get_semantic_synonym(token.text, token.tag_)
            processed_words.append(synonym)
        else:
            processed_words.append(token.text)

    reconstructed_text = " ".join(processed_words)
    reconstructed_text = re.sub(r'\s+([.,!?;:])', r'\1', reconstructed_text)

    # تقسيم الجمل بالاعتماد على ذكاء spaCy بدلاً من NLTK المنهار
    sentences = [s.text.strip() for s in nlp(reconstructed_text).sents]
    humanized_sentences = []
    last_transition = None

    for i, sentence in enumerate(sentences):
        if i > 0 and random.random() < p_trans:
            available_transitions = [t for t in ACADEMIC_TRANSITIONS if t != last_transition]
            chosen_transition = random.choice(available_transitions)
            sentence = f"{chosen_transition}, {sentence[0].lower()}{sentence[1:]}"
            last_transition = chosen_transition
            
        humanized_sentences.append(sentence)

    final_text = " ".join(humanized_sentences)

    for i, citation in enumerate(citations):
        final_text = final_text.replace(f"__CITATION_{i}__", citation)
        
    final_text = re.sub(r'\s+__CITATION_\d+__\s+', ' ', final_text)
    for i, citation in enumerate(citations):
        final_text = final_text.replace(f"__CITATION_{i}__", citation)

    return final_text.strip()

@app.post("/humanize")
async def humanize_endpoint(request: HumanizeRequest):
    # استخدام تحليل spaCy لحساب الإحصائيات بدقة وتفادي أخطاء حزم NLTK
    doc_orig = nlp(request.text)
    orig_words = len(request.text.split())
    orig_sentences = len(list(doc_orig.sents))
    
    result_text = advanced_humanizer(request.text, request.p_syn, request.p_trans)
    
    doc_new = nlp(result_text)
    new_words = len(result_text.split())
    new_sentences = len(list(doc_new.sents))
    
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
    return {"status": "healthy", "engine": "spaCy + WordNet Expert"}
