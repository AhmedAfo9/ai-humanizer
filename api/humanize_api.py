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

app = FastAPI(title="Advanced Academic AI Text Humanizer API", version="2.8")

# --- إعدادات الـ CORS المستقرة والمفتوحة ---
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

ACADEMIC_TRANSITIONS = [
    "Furthermore", "Moreover", "In addition", "Consequently", 
    "Therefore", "On the other hand", "Notably", "In this context"
]

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


# --- 🧠 خوارزمية التحويل الهيكلي والأنسنة النحوية الفائقة المحدثة ---
def advanced_structural_reshaper(sent_text):
    doc = nlp(sent_text)
    text = sent_text.strip()
    
    # 1. 🔄 نقل الظروف (Adverb Shifting) - آمن للغاية وممتاز لكسر الـ N-grams
    # مثال: "The system analyzed the data thoroughly." -> "Thoroughly, the system analyzed the data."
    adverbs = [tok for tok in doc if tok.pos_ == "ADV" and tok.text.lower().endswith("ly")]
    if adverbs:
        adv = adverbs[0]
        if adv.text.lower() not in ["only", "really", "very", "simply", "just"]:
            adv_text = adv.text
            # إزالة الظرف القديم وإعادة بناء الجملة بذكاء
            cleaned_text = re.sub(rf'\b{adv_text}\b', '', text).strip()
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = re.sub(r'\s+([.,!?;:])', r'\1', cleaned_text)
            
            # ضبط حالة الأحرف الاستهلالية للجملة الجديدة
            if cleaned_text[0].isupper() and not cleaned_text.split()[0].lower() in ['i']:
                cleaned_text = cleaned_text[0].lower() + cleaned_text[1:]
            return f"{adv_text.capitalize()}, {cleaned_text}"

    # 2. 🔄 قلب الجمل الشرطية والسببية (Clause Inversion)
    clause_match = re.search(r'(\s+)(because|although|since|while|if|when|though)\s+([^.,!?;]+)', text, re.IGNORECASE)
    if clause_match:
        conj = clause_match.group(2)
        clause_content = clause_match.group(3)
        parts = text.split(clause_match.group(0))
        if len(parts) == 2 and parts[0].strip():
            main_clause = parts[0].strip()
            sub_clause = conj + " " + clause_content.strip() + parts[1].strip()
            
            sub_clause = sub_clause[0].upper() + sub_clause[1:]
            if main_clause[0].isupper() and not main_clause.split()[0].lower() in ['i', 'the', 'this', 'it', 'a', 'an']:
                main_clause = main_clause[0].lower() + main_clause[1:]
                
            return f"{sub_clause}, {main_clause.rstrip('.')}"

    # 3. 🔄 المبني للمجهول الأكاديمي الشائع جداً
    academic_patterns = {
        r'^This study demonstrates that\s+(.+)' : r'It is demonstrated by this study that \1',
        r'^The researchers found that\s+(.+)' : r'It was found by the researchers that \1',
        r'^The paper analyzes how\s+(.+)' : r'How \1 is analyzed by this paper',
        r'^This suggests that\s+(.+)' : r'It is suggested by this that \1',
        r'^The results show that\s+(.+)' : r'It is shown by the results that \1'
    }
    for pattern, repl in academic_patterns.items():
        if re.match(pattern, text, re.IGNORECASE):
            return re.sub(pattern, repl, text, flags=re.IGNORECASE)

    if "plays a crucial role in" in text.lower():
        return re.sub(r'(.+?)\s+plays a crucial role in\s+(.+)', r'A crucial role is played by \1 in \2', text, flags=re.IGNORECASE)

    # 4. 🛡️ شبكة الأمان الميكروسكوبية (Fallback Safety Net)
    # إذا لم يطبق أي تعديل هيكلي، نستبدل كلمة "فاضحة للـ AI" واحدة فقط لكسر مطابقة النص وحمايته!
    ai_watermarks = {
        "moreover": "furthermore",
        "furthermore": "in addition",
        "delve": "explore",
        "testament": "proof",
        "tapestry": "complexity",
        "utilize": "use",
        "crucial": "essential",
        "essential": "important",
        "significant": "notable",
        "notable": "remarkable",
        "consequently": "therefore",
        "therefore": "thus",
        "thus": "hence"
    }
    words = text.split()
    changed = False
    for idx, w in enumerate(words):
        clean_w = w.lower().strip(".,!?;:")
        if clean_w in ai_watermarks:
            repl = ai_watermarks[clean_w]
            if w.istitle():
                repl = repl.title()
            words[idx] = w.replace(clean_w, repl)
            changed = True
            break # نكتفي بكلمة واحدة ليبقى النص الأصلي 99% كما هو!
            
    if changed:
        return " ".join(words)

    return text


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
    
    if request.use_passive:
        sentences = [s.text.strip() for s in doc_orig.sents]
        reshaped_sentences = []
        
        for sent in sentences:
            reshaped_text = advanced_structural_reshaper(sent)
            reshaped_sentences.append(reshaped_text)
            
        result_text = " ".join(reshaped_sentences)
    else:
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
    return {"status": "healthy", "engine": "spaCy Structural Reshaper v2.8"}
