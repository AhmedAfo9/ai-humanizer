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

app = FastAPI(title="Advanced Academic AI Text Humanizer API", version="3.0")

# --- إعدادات الـ CORS المفتوحة والمستقرة لجميع المنصات ---
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


# --- 🧠 خوارزمية التحويل الهيكلي والأنسنة النحوية الفائقة v3.0 (سحق كواشف الـ AI) ---
def advanced_structural_reshaper(sent_text):
    text = sent_text.strip()
    if not text:
        return ""
        
    # 1. 🛡️ حفظ وعزل علامة الترقيم في نهاية الجملة تماماً لمنع الأخطاء اللغوية
    ending_punc = "."
    if text[-1] in [".", "!", "?", ";"]:
        ending_punc = text[-1]
        text = text[:-1].strip()
        
    # 2. 🛡️ حفظ وعزل العبارات الاستهلالية وتطبيعها لغوياً لكسر بصمة الـ AI
    intro_phrase = ""
    intro_match = re.match(r'^(In addition|Furthermore|Moreover|Therefore|Consequently|However|On the other hand|Thus|Notably|As a result|To begin with|In this context),\s+', text, re.IGNORECASE)
    if intro_match:
        intro_phrase = intro_match.group(0)
        text = text[len(intro_phrase):].strip()
        
    # ترقية وتغيير العبارة الاستهلالية الفاضحة ببديل أكاديمي نادر ومقاوم للكواشف
    if intro_phrase:
        intro_lower = intro_phrase.lower().strip(", ")
        intro_map = {
            "therefore": "Thus, ",
            "furthermore": "Moreover, ",
            "moreover": "In addition, ",
            "in addition": "Additionally, ",
            "consequently": "As a consequence, ",
            "however": "Nonetheless, ",
            "thus": "Hence, "
        }
        if intro_lower in intro_map:
            intro_phrase = intro_map[intro_lower]

    structured = False
    
    # 3. 🔄 خوارزمية قلب الأدوار وصياغة المبني للمجهول (Active to Passive)
    # النمط A: "X is the key to Y" -> "Y is fundamentally driven by X"
    if not structured and re.search(r'^(.+?)\s+is\s+the\s+key\s+to\s+(.+)$', text, re.IGNORECASE):
        text = re.sub(r'^(.+?)\s+is\s+the\s+key\s+to\s+(.+)$', r'\2 is fundamentally driven by \1', text, flags=re.IGNORECASE)
        structured = True
        
    # النمط B: "X gives/provides people Y" -> "People are provided Y by X"
    if not structured and re.search(r'^(.+?)\s+(gives|provides)\s+people\s+(.+)$', text, re.IGNORECASE):
        text = re.sub(r'^(.+?)\s+(gives|provides)\s+people\s+(.+)$', r'People are provided \3 by \1', text, flags=re.IGNORECASE)
        structured = True
        
    # النمط C: "X helps them Y" -> "They are helped by X to Y"
    if not structured and re.search(r'^(.+?)\s+(also\s+)?helps\s+them\s+(.+)$', text, re.IGNORECASE):
        also_part = re.search(r'^(.+?)\s+(also\s+)?helps\s+them\s+(.+)$', text, re.IGNORECASE).group(2) or ""
        text = re.sub(r'^(.+?)\s+(also\s+)?helps\s+them\s+(.+)$', rf'They are {also_part}helped by \1 to \3', text, flags=re.IGNORECASE)
        structured = True
        
    # النمط D: "X opens the door to Y" -> "The door to Y is opened by X"
    if not structured and re.search(r'^(.+?)\s+opens\s+the\s+door\s+to\s+(.+)$', text, re.IGNORECASE):
        text = re.sub(r'^(.+?)\s+opens\s+the\s+door\s+to\s+(.+)$', r'The door to \2 is opened by \1', text, flags=re.IGNORECASE)
        structured = True

    # النمط E: قلب جمل الشرط والسببية (Inversion)
    if not structured:
        clause_match = re.search(r'(\s+)(because|although|since|while|if|when|though)\s+([^.,!?;]+)$', text, re.IGNORECASE)
        if clause_match:
            conj = clause_match.group(2)
            clause_content = clause_match.group(3)
            parts = text.split(clause_match.group(0))
            if len(parts) == 2 and parts[0].strip():
                main_clause = parts[0].strip()
                sub_clause = conj + " " + clause_content.strip()
                
                if main_clause[0].isupper() and not main_clause.split()[0].lower() in ['i']:
                    main_clause = main_clause[0].lower() + main_clause[1:]
                    
                text = f"{sub_clause}, {main_clause}"
                structured = True

    # 4. 🧩 الأنسنة والتبديل العباري المتقدم (Phrase Mapping) لكسر تتابع كلمات الـ AI الرتيبة
    phrase_replacements = {
        r'\bpeople who continue learning\b': 'those continuing to learn',
        r'\bcan adapt to\b': 'are highly capable of adapting to',
        r'\bachieve their goals\b': 'attaining their core objectives',
        r'\bimportant investments\b': 'vital endeavors',
        r'\banyone can make\b': 'one can undertake',
        r'\bfor a brighter future\b': 'toward a more promising outlook',
        r'\blearn to\b': 'acquire the ability to',
        r'\bsolve problems\b': 'address complex challenges',
        r'\bcommunicate effectively\b': 'interact with high efficacy',
        r'\bknowledge and skills\b': 'foundational expertise and competencies'
    }
    for pat, repl in phrase_replacements.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # 5. 🔄 نقل الظروف (Adverb Shifting) - يطبق فقط إذا لم تتغير البنية مسبقاً
    if not structured:
        doc = nlp(text)
        adverbs = [tok for tok in doc if tok.pos_ == "ADV" and tok.text.lower().endswith("ly")]
        if adverbs:
            adv = adverbs[0]
            if adv.text.lower() not in ["only", "really", "very", "simply", "just"]:
                adv_text = adv.text
                cleaned_text = re.sub(rf'\b{adv_text}\b', '', text).strip()
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                if cleaned_text[0].isupper() and not cleaned_text.split()[0].lower() in ['i']:
                    cleaned_text = cleaned_text[0].lower() + cleaned_text[1:]
                text = f"{adv_text.capitalize()}, {cleaned_text}"
                structured = True

    # 6. 🛡️ شبكة الأمان وحارس الكلمات الفاضحة (Case-Insensitive Fixed)
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
        clean_w = re.sub(r'[^a-zA-Z]', '', w).lower()
        if clean_w in ai_watermarks:
            repl = ai_watermarks[clean_w]
            if w[0].isupper():
                repl = repl.capitalize()
            # استخدام ريجكس محدد الحدود لتبديل الكلمة والاحتفاظ بأي فاصلة ملتصقة بها
            words[idx] = re.sub(rf'\b{clean_w}\b', repl, w, flags=re.IGNORECASE)
            changed = True
            
    if changed:
        text = " ".join(words)

    # 7. 🎬 إعادة تجميع وترتيب الجملة وتصحيح الأحرف الكبيرة
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
        
    final_sentence = f"{intro_phrase}{text}{ending_punc}"
    
    # تنظيف وتلميع علامات الترقيم لمنع مشكلة تكرار "., " أو ",. " نهائياً
    final_sentence = re.sub(r'\s+', ' ', final_sentence)
    final_sentence = re.sub(r'\.+', '.', final_sentence)
    final_sentence = final_sentence.replace(".,", ",")
    final_sentence = final_sentence.replace(",.", ",")
    final_sentence = final_sentence.replace(", ,", ",")
    
    return final_sentence


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
    return {"status": "healthy", "engine": "spaCy Structural Reshaper v3.0"}
