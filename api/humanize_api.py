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

app = FastAPI(title="Ultimate Absolute Academic Humanizer API", version="7.0")

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

ACADEMIC_TRANSITIONS = [
    "Furthermore", "Moreover", "In addition", "Consequently", 
    "Therefore", "On the other hand", "Notably", "In this context"
]

IRREGULAR_PLURALS = {
    "analysis": "analyses", "hypothesis": "hypotheses", "thesis": "theses",
    "criterion": "criteria", "phenomenon": "phenomena", "diagnosis": "diagnoses", "datum": "data"
}

# الأفعال التركيبية والمساعدة المحظور تبديلها نهائياً لحماية القواعد الأزمنية
PROTECTED_VERBS = {
    "be", "is", "am", "are", "was", "were", "been", "being", 
    "have", "has", "had", "do", "does", "did", "can", "could", 
    "will", "would", "shall", "should", "may", "might", "must"
}

# --- 🛡️ دالة حارس أدوات النكرة الصوتي المتقدم v7.0 ---
def fix_indefinite_articles(text):
    # 1. التبديل القياسي الأولي: a تحول إلى an أمام الحروف المتحركة
    text = re.sub(r'\b([Aa])\s+([aeiouAEIOU][a-zA-Z]*)', r'\1n \2', text)
    
    # 2. الاستثناء الصوتي الحاسم: الكلمات المبتدئة بصوت ساكن رغم الحرف المتحرك (مثل unique, university)
    # يتم إرجاعها تلقائياً إلى a لضمان الكمال النحوي 100%
    text = re.sub(r'\b([Aa])n\s+(unique|university|universal|union|unit|uniform|unilateral|one)\b', r'\1 \2', text, flags=re.IGNORECASE)
    
    # 3. التبديل القياسي العكسي: an تحول إلى a أمام الحروف الساكنة
    text = re.sub(r'\b([Aa])n\s+([^aeiouAEIOU][a-zA-Z]*)', r'\1 \2', text)
    return text

# --- 🛠️ دالة التصريف النحوي التلقائي الذكي ---
def smart_inflect(synonym_word, original_token):
    tag = original_token.tag_
    word = synonym_word.lower()
    
    if tag in ["NNS", "NNPS"] and word in IRREGULAR_PLURALS:
        word = IRREGULAR_PLURALS[word]
    elif tag in ["NNS", "NNPS"]:
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            word = word[:-1] + "ies"
        elif any(word.endswith(x) for x in ["s", "sh", "ch", "x", "z"]):
            if not word.endswith("es"): word = word + "es"
        else:
            word = word + "s"
    elif tag == "VBG":
        if word.endswith("e") and not any(word.endswith(x) for x in ["ee", "oe", "ye"]):
            word = word[:-1] + "ing"
        elif word.endswith("ie"):
            word = word[:-2] + "ying"
        else:
            word = word + "ing"
    elif tag in ["VBD", "VBN"]:
        if word.endswith("e"):
            word = word + "d"
        elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            word = word[:-1] + "ied"
        else:
            word = word + "ed"
    elif tag == "VBZ":
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            word = word[:-1] + "ies"
        elif any(word.endswith(x) for x in ["s", "sh", "ch", "x", "z", "o"]):
            if not word.endswith("es"): word = word + "es"
        else:
            word = word + "s"

    if original_token.text.istitle(): return word.title()
    elif original_token.text.isupper(): return word.upper()
    return word

# --- 🧠 مصفوفة البدائل الأكاديمية المحدثة بالكامل (v7.0 منيعة الأخطاء) ---
def get_perfect_context_substitution(token, doc, idx):
    text_lower = token.text.lower()
    
    # حماية فورية للأفعال المساعدة والبنائية من التبديل العشوائي لمنع "be studying"
    if text_lower in PROTECTED_VERBS or token.pos_ == "AUX":
        return token.text
    
    if text_lower == "hard":
        if idx > 0 and doc[idx-1].pos_ == "VERB" and doc[idx-1].text.lower() in ["studying", "study", "working", "work", "tries"]:
            return "diligently"
        return "challenging"
        
    if text_lower in ["methodologies", "methodology"]:
        return "methodological frameworks" if token.tag_ in ["NNS", "NNPS"] else "methodological framework"

    # تصفية الكلمات الأساسية لضمان مفردات طبيعية بليغة تفتت بصمة الـ AI
    fixed_matrix = {
        "student": "learner", "students": "learners",
        "utilize": "use", "utilizes": "employs", "utilized": "employed", "utilizing": "employing",
        "achieve": "attain", "achieved": "attained", "achieves": "attains", "achieving": "attaining",
        "goal": "objective", "goals": "objectives", "opportunity": "avenue", "opportunities": "avenues",
        "provide": "offer", "provides": "offers", "provided": "offered", "providing": "offering",
        "conduct": "undertake", "conducted": "undertaken", "conducts": "undertakes", "conducting": "undertaking",
        "researcher": "investigator", "researchers": "investigators", "study": "inquiry", "studies": "inquiries",
        "important": "pivotal", "crucial": "paramount", "significant": "notable", "complex": "intricate", "modern": "contemporary"
    }
    
    if text_lower in fixed_matrix:
        res = fixed_matrix[text_lower]
        if token.text.istitle(): return res.capitalize()
        return res
    return None

def get_contextual_synonym(token, doc, idx):
    perfect_match = get_perfect_context_substitution(token, doc, idx)
    if perfect_match: return perfect_match
    
    # إذا كانت الكلمة محمية كفعل مساعد، أرجعها فوراً
    if token.text.lower() in PROTECTED_VERBS or token.pos_ == "AUX":
        return token.text
        
    word = token.text
    pos_tag = token.tag_
    wn_tag = None
    if pos_tag.startswith("NN"): wn_tag = wordnet.NOUN
    elif pos_tag.startswith("VB"): wn_tag = wordnet.VERB
    elif pos_tag.startswith("JJ"): wn_tag = wordnet.ADJ
    elif pos_tag.startswith("RB"): wn_tag = wordnet.ADV
    
    if not wn_tag: return word

    try:
        orig_synsets = wordnet.synsets(word, pos=wn_tag)
        if not orig_synsets: return word
        orig_syn = orig_synsets[0]
        
        candidates = []
        for synset in wordnet.synsets(word, pos=wn_tag):
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower() and name.isalpha():
                    cand_synsets = wordnet.synsets(lemma.name(), pos=wn_tag)
                    if cand_synsets:
                        similarity = orig_syn.path_similarity(cand_synsets[0])
                        if similarity and similarity >= 0.45:
                            candidates.append(name)
                            
        if candidates:
            chosen = random.choice(candidates)
            return smart_inflect(chosen, token)
    except Exception:
        return word
    return word

def inject_smart_transition(sentence_text, transition):
    doc = nlp(sentence_text)
    nsubj_token = None
    for token in doc:
        if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PRON", "PROPN"]:
            nsubj_token = token
            break
            
    if nsubj_token and random.random() < 0.4:
        parts = sentence_text.split(nsubj_token.text, 1)
        if len(parts) == 2:
            left = parts[0] + nsubj_token.text
            right = parts[1].strip()
            if right and right[0].isupper() and not right.split()[0].lower() in ['i']:
                right = right[0].lower() + right[1:]
            return f"{left}, {transition.lower()}, {right}"
            
    return f"{transition}, {sentence_text[0].lower()}{sentence_text[1:]}"

def advanced_structural_reshaper(sent_text):
    text = sent_text.strip()
    if not text: return ""
        
    ending_punc = "."
    if text[-1] in [".", "!", "?", ";"]:
        ending_punc = text[-1]
        text = text[:-1].strip()
        
    intro_phrase = ""
    intro_match = re.match(r'^(In addition|Furthermore|Moreover|Therefore|Consequently|However|On the other hand|Thus|Notably|As a result|To begin with|In this context|Additionally),\s+', text, re.IGNORECASE)
    if intro_match:
        intro_phrase = intro_match.group(0)
        text = text[len(intro_phrase):].strip()
        
    if intro_phrase:
        intro_lower = intro_phrase.lower().strip(", ")
        intro_map = {
            "therefore": "Thus, ", "furthermore": "Moreover, ", "moreover": "In addition, ",
            "in addition": "Additionally, ", "consequently": "As a consequence, ",
            "however": "Nonetheless, ", "thus": "Hence, "
        }
        if intro_lower in intro_map: intro_phrase = intro_map[intro_lower]

    structured = False
    
    if not structured and re.search(r'^(.+?)\s+is\s+the\s+key\s+to\s+(.+)$', text, re.IGNORECASE):
        text = re.sub(r'^(.+?)\s+is\s+the\s+key\s+to\s+(.+)$', r'\2 is fundamentally driven by \1', text, flags=re.IGNORECASE)
        structured = True
        
    if not structured and re.search(r'^(.+?)\s+(gives|provides)\s+people\s+(.+)$', text, re.IGNORECASE):
        text = re.sub(r'^(.+?)\s+(gives|provides)\s+people\s+(.+)$', r'People are provided \3 by \1', text, flags=re.IGNORECASE)
        structured = True
        
    if not structured and re.search(r'^(.+?)\s+(also\s+)?helps\s+them\s+(.+)$', text, re.IGNORECASE):
        also_part = re.search(r'^(.+?)\s+(also\s+)?helps\s+them\s+(.+)$', text, re.IGNORECASE).group(2) or ""
        text = re.sub(r'^(.+?)\s+(also\s+)?helps\s+them\s+(.+)$', rf'They are {also_part}helped by \1 to \3', text, flags=re.IGNORECASE)
        structured = True
        
    if not structured and re.search(r'^(.+?)\s+opens\s+the\s+door\s+to\s+(.+)$', text, re.IGNORECASE):
        text = re.sub(r'^(.+?)\s+opens\s+the\s+door\s+to\s+(.+)$', r'The door to \2 is opened by \1', text, flags=re.IGNORECASE)
        structured = True

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

    ai_watermarks = {
        "moreover": "furthermore", "furthermore": "in addition", "delve": "explore",
        "testament": "proof", "tapestry": "complexity", "utilize": "use",
        "crucial": "essential", "essential": "important", "significant": "notable",
        "notable": "remarkable", "consequently": "therefore", "therefore": "thus", "thus": "hence"
    }
    words = text.split()
    changed = False
    for idx, w in enumerate(words):
        clean_w = re.sub(r'[^a-zA-Z]', '', w).lower()
        if clean_w in ai_watermarks:
            repl = ai_watermarks[clean_w]
            if w[0].isupper(): repl = repl.capitalize()
            words[idx] = re.sub(rf'\b{clean_w}\b', repl, w, flags=re.IGNORECASE)
            changed = True
            
    if changed: text = " ".join(words)

    text = text.strip()
    if text: text = text[0].upper() + text[1:]
        
    final_sentence = f"{intro_phrase}{text}{ending_punc}"
    
    final_sentence = re.sub(r'\s+', ' ', final_sentence)
    final_sentence = re.sub(r'\.+', '.', final_sentence)
    final_sentence = final_sentence.replace(".,", ",")
    final_sentence = final_sentence.replace(",.", ",")
    final_sentence = final_sentence.replace("by It", "by it")
    final_sentence = final_sentence.replace("by Education", "by education")
    final_sentence = final_sentence.replace("to It", "to it")
    final_sentence = final_sentence.replace(" , ", ", ")
    final_sentence = final_sentence.replace(" .", ".")
    
    return fix_indefinite_articles(final_sentence)


def advanced_humanizer(text, p_syn, p_trans):
    citation_pattern = r'(\([A-Za-z\s\.\,]+,\s+\d{4}\)|\[\d+\])'
    citations = re.findall(citation_pattern, text)
    
    for i, citation in enumerate(citations):
        text = text.replace(citation, f" __CITATION_{i}__ ")

    doc = nlp(text)
    sentences = list(doc.sents)
    humanized_sentences = []
    last_transition = None

    for i, sent in enumerate(sentences):
        processed_words = []
        protected_entities = [ent.text.lower() for ent in sent.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]
        
        for idx, token in enumerate(sent):
            if "__CITATION_" in token.text or token.text.lower() in protected_entities or token.is_punct or token.is_digit:
                processed_words.append(token.text)
                continue
            
            perfect_sub = get_perfect_context_substitution(token, sent, idx)
            
            # حظر تبديل الأفعال المساعدة حتى لو تحرك الشريط عشوائياً
            if token.text.lower() in PROTECTED_VERBS or token.pos_ == "AUX":
                processed_words.append(token.text)
                continue
                
            if random.random() < p_syn or perfect_sub:
                synonym = perfect_sub if perfect_sub else get_contextual_synonym(token, sent, idx)
                processed_words.append(synonym)
            else:
                processed_words.append(token.text)

        reconstructed_sent = " ".join(processed_words)
        # إصلاح دالة المعالجة الموضعية للجملة (تم التطهير التام من الـ Bug القديم)
        reconstructed_sent = re.sub(r'\s+([.,!?;:])', r'\1', reconstructed_sent)
        
        if i > 0 and random.random() < p_trans:
            available_transitions = [t for t in ACADEMIC_TRANSITIONS if t != last_transition]
            chosen_transition = random.choice(available_transitions)
            reconstructed_sent = inject_smart_transition(reconstructed_sent, chosen_transition)
            last_transition = chosen_transition
            
        humanized_sentences.append(reconstructed_sent)

    final_text = " ".join(humanized_sentences)
    for i, citation in enumerate(citations):
        final_text = final_text.replace(f"__CITATION_{i}__", citation)
        
    final_text = re.sub(r'\s+__CITATION_\d+__\s+', ' ', final_text)
    for i, citation in enumerate(citations):
        final_text = final_text.replace(f"__CITATION_{i}__", citation)

    final_text = re.sub(r'\s+,\s+', ', ', final_text)
    final_text = final_text.replace(" ,", ",")
    final_text = re.sub(r'\s+\.', '.', final_text)
    
    return fix_indefinite_articles(final_text.strip())


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
    return {"status": "healthy", "engine": "Ultimate Absolute Engine v7.0"}
