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

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

app = FastAPI(title="Advanced Academic AI Text Humanizer API", version="2.5")

# --- إعدادات الـ CORS المفتوحة والمستقرة لجميع المنصات ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- هيكل البيانات المستقبلة المطور لدعم مفتاح الـ Passive Voice ---
class HumanizeRequest(BaseModel):
    text: str
    p_syn: Optional[float] = 0.2
    p_trans: Optional[float] = 0.2
    preserve_linebreaks: Optional[bool] = True
    use_passive: Optional[bool] = False  # المتغير الجديد للتحكم بالميزة الهيكلية

# --- قائمة الروابط الأكاديمية (تستخدم في النمط التقليدي) ---
ACADEMIC_TRANSITIONS = [
    "Furthermore", "Moreover", "In addition", "Consequently", 
    "Therefore", "On the other hand", "Notably", "In this context"
]

# --- دالة جلب المترادفات (تستخدم في النمط التقليدي) ---
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


# --- 🧠 خوارزمية التحويل الهيكلي والأنسنة النحوية الفائقة (Syntactic Reshaper) ---
def advanced_structural_reshaper(sent_text):
    text = sent_text.strip()
    
    # 1. كسر نمط الـ AI في الجمل الشرطية والسببية عبر قلب العبارات (Clause Inversion)
    # يحول: "The results are valid because the sample size was huge."
    # إلى: "Because the sample size was huge, the results are valid."
    clause_match = re.search(r'(\s+)(because|although|since|while|if|when|though)\s+([^.,!?;]+)', text, re.IGNORECASE)
    if clause_match:
        conj = clause_match.group(2)
        clause_content = clause_match.group(3)
        parts = text.split(clause_match.group(0))
        if len(parts) == 2 and parts[0].strip():
            main_clause = parts[0].strip()
            sub_clause = conj + " " + clause_content.strip() + parts[1].strip()
            
            # ضبط حالة الأحرف الكبيرة والصغيرة (Capitalization) بذكاء بشرى
            sub_clause = sub_clause[0].upper() + sub_clause[1:]
            if main_clause[0].isupper() and not main_clause.split()[0].lower() in ['i', 'the', 'this', 'it', 'a', 'an']:
                main_clause = main_clause[0].lower() + main_clause[1:]
                
            return f"{sub_clause}, {main_clause.rstrip('.')}"

    # 2. تحويل العبارات الاستهلالية المكشوفة للـ AI إلى صيغ أكاديمية مبنية للمجهول (Passive Framing)
    text = re.sub(r'^This study demonstrates that\s+(.+)', r'It is demonstrated by this study that \1', text, flags=re.IGNORECASE)
    text = re.sub(r'^The author argues that\s+(.+)', r'It is argued by the author that \1', text, flags=re.IGNORECASE)
    text = re.sub(r'^The results show that\s+(.+)', r'It is shown by the results that \1', text, flags=re.IGNORECASE)
    text = re.sub(r'^Researchers found that\s+(.+)', r'It was found by researchers that \1', text, flags=re.IGNORECASE)
    
    # 3. صياغة قلب الأدوار لـ الأفعال النمطية الشائعة جداً في نصوص الذكاء الاصطناعي
    if "plays a crucial role in" in text.lower():
        text = re.sub(r'(.+?)\s+plays a crucial role in\s+(.+)', r'A crucial role is played by \1 in \2', text, flags=re.IGNORECASE)
    elif "plays an important role in" in text.lower():
        text = re.sub(r'(.+?)\s+plays an important role in\s+(.+)', r'An important role is played by \1 in \2', text, flags=re.IGNORECASE)

    return text


# --- دالة الأنسنة التقليدية بالمفردات والروابط ---
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
    doc_orig = nlp(request.text)
    orig_words = len(request.text.split())
    orig_sentences = len(list(doc_orig.sents))
    
    # --- المسار الحرج: التحقق مما إذا كان تم تفعيل الـ Passive Voice الهيكلي ---
    if request.use_passive:
        sentences = [s.text.strip() for s in doc_orig.sents]
        reshaped_sentences = []
        
        for index, sent in enumerate(sentences):
            # تمرير الجمل عبر خوارزمية إعادة الهيكلة النحوية
            reshaped_text = advanced_structural_reshaper(sent)
            reshaped_sentences.append(reshaped_text)
            
        result_text = " ".join(reshaped_sentences)
    else:
        # النمط التقليدي (تعديل المفردات والروابط بناءً على أشرطة السحب)
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
    return {"status": "healthy", "engine": "spaCy Structural Syntactic Reshaper v2.5"}
