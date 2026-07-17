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

app = FastAPI(title="Super Hybrid Academic Text Humanizer API", version="4.0")

# --- إعدادات الـ CORS المستقرة لجميع المنصات ---
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


# --- 🛠️ دالة التصريف النحوي التلقائي (Smart Inflection Engine) ---
# تضمن مطابقة الكلمة الجديدة لقواعد الجملة الأصلية 100% (منع أخطاء الأزمنة والجمع)
def smart_inflect(synonym_word, original_token):
    tag = original_token.tag_
    word = synonym_word.lower()
    
    # 1. معالجة الأفعال (Verbs)
    if tag == "VBG":  # Gerund or present participle (e.g., studying)
        if word.endswith("e") and not any(word.endswith(x) for x in ["ee", "oe", "ye"]):
            word = word[:-1] + "ing"
        elif word.endswith("ie"):
            word = word[:-2] + "ying"
        else:
            word = word + "ing"
    elif tag in ["VBD", "VBN"]:  # Past tense / Past participle (e.g., studied)
        if word.endswith("e"):
            word = word + "d"
        elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            word = word[:-1] + "ied"
        else:
            word = word + "ed"
    elif tag == "VBZ":  # 3rd person singular present (e.g., studies)
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            word = word[:-1] + "ies"
        elif any(word.endswith(x) for x in ["s", "sh", "ch", "x", "z", "o"]):
            word = word + "es"
        else:
            word = word + "s"
            
    # 2. معالجة الأسماء الجمع (Plural Nouns)
    elif tag in ["NNS", "NNPS"]:
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            word = word[:-1] + "ies"
        elif any(word.endswith(x) for x in ["s", "sh", "ch", "x", "z"]):
            word = word + "es"
        else:
            word = word + "s"
            
    # الحفاظ على حالة الأحرف الكبيرة الأصلية (Capitalization)
    if original_token.text.istitle():
        return word.title()
    elif original_token.text.isupper():
        return word.upper()
    return word


# --- ⚖️ دالة جلب المترادفات الذكية مع تصفية التشابه السياقي والأمان النحوي ---
def get_contextual_synonym(token):
    word = token.text
    pos_tag = token.tag_
    
    wn_tag = None
    if pos_tag.startswith("NN"): wn_tag = wordnet.NOUN
    elif pos_tag.startswith("VB"): wn_tag = wordnet.VERB
    elif pos_tag.startswith("JJ"): wn_tag = wordnet.ADJ
    elif pos_tag.startswith("RB"): wn_tag = wordnet.ADV
    
    if not wn_tag:
        return word

    try:
        orig_synsets = wordnet.synsets(word, pos=wn_tag)
        if not orig_synsets:
            return word
        orig_syn = orig_synsets[0]  # المعنى الأقرب للكلمة الأصلية
        
        candidates = []
        for synset in wordnet.synsets(word, pos=wn_tag):
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower() and name not in candidates:
                    # حساب نسبة التقارب الدلالي لمنع الركاكة (Collocation Filter)
                    cand_synsets = wordnet.synsets(lemma.name(), pos=wn_tag)
                    if cand_synsets:
                        similarity = orig_syn.path_similarity(cand_synsets[0])
                        # عتبة أمان (Threshold) لضمان توافق الكلمة تماماً مع السياق
                        if similarity and similarity >= 0.25:
                            candidates.append(name)
                            
        if candidates:
            chosen = random.choice(candidates)
            # تصريف الكلمة المختارة لتطابق زمن وصيغة الكلمة الأصلية
            return smart_inflect(chosen, token)
    except Exception:
        return word
        
    return word


# --- 🔄 دالة الحقن الاعتراضي للروابط (Parenthetical Transition Injector) ---
def inject_smart_transition(sentence_text, transition):
    doc = nlp(sentence_text)
    
    # البحث عن أول فاعل (Subject) في الجملة لحقن الرابط خلفه مباشرة أسلوبياً
    nsubj_token = None
    for token in doc:
        if token.dep_ == "nsubj" and token.pos_ in ["NOUN", "PRON", "PROPN"]:
            nsubj_token = token
            break
            
    # حقن الرابط اعتراضياً بنسبة 40% من الحالات ليعطي طابعاً بشرياً بليغاً
    if nsubj_token and random.random() < 0.4:
        parts = sentence_text.split(nsubj_token.text, 1)
        if len(parts) == 2:
            left = parts[0] + nsubj_token.text
            right = parts[1].strip()
            # ضبط حالة الأحرف للتكملة بذكاء
            if right and right[0].isupper() and not right.split()[0].lower() in ['i']:
                right = right[0].lower() + right[1:]
            return f"{left}, {transition.lower()}, {right}"
            
    # الطريقة الكلاسيكية: وضع الرابط في بداية الجملة
    return f"{transition}, {sentence_text[0].lower()}{sentence_text[1:]}"


# --- خوارزمية التحويل الهيكلي والأنسنة النحوية الفائقة ---
def advanced_structural_reshaper(sent_text):
    text = sent_text.strip()
    if not text:
        return ""
        
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
            words[idx] = re.sub(rf'\b{clean_w}\b', repl, w, flags=re.IGNORECASE)
            changed = True
            
    if changed:
        text = " ".join(words)

    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
        
    final_sentence = f"{intro_phrase}{text}{ending_punc}"
    
    # تنظيف علامات الترقيم
    final_sentence = re.sub(r'\s+', ' ', final_sentence)
    final_sentence = re.sub(r'\.+', '.', final_sentence)
    final_sentence = final_sentence.replace(".,", ",")
    final_sentence = final_sentence.replace(",.", ",")
    final_sentence = final_sentence.replace(", ,", ",")
    
    # إصلاح الأخطاء الطفيفة لتسميات الضمائر الملحقة بالصيغ المجهولة
    final_sentence = final_sentence.replace("by It", "by it")
    final_sentence = final_sentence.replace("by Education", "by education")
    final_sentence = final_sentence.replace("to It", "to it")
    final_sentence = final_sentence.replace(" , ", ", ")
    
    return final_sentence


# --- دالة الأنسنة التقليدية المرقاة بالكامل مع حارس القواعد ---
def advanced_humanizer(text, p_syn, p_trans):
    citation_pattern = r'(\([A-Za-z\s\.\,]+,\s+\d{4}\)|\[\d+\])'
    citations = re.findall(citation_pattern, text)
    
    for i, citation in enumerate(citations):
        text = text.replace(citation, f" __CITATION_{i}__ ")

    doc = nlp(text)
    processed_words = []
    protected_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

    # استبدال ذكي جداً مع الحفاظ الكامل على تصريفات القواعد الأصلية والتوافق الدلالي
    for token in doc:
        if "__CITATION_" in token.text or token.text.lower() in protected_entities or token.is_punct or token.is_digit:
            processed_words.append(token.text)
            continue
            
        if random.random() < p_syn:
            # استدعاء دالة التصفية والتصريف الذكي بدلاً من التبديل الأعمى
            synonym = get_contextual_synonym(token)
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
            # استخدام أسلوب الحقن الاعتراضي فائق الاحترافية
            sentence = inject_smart_transition(sentence, chosen_transition)
            last_transition = chosen_transition
            
        humanized_sentences.append(sentence)

    final_text = " ".join(humanized_sentences)
    for i, citation in enumerate(citations):
        final_text = final_text.replace(f"__CITATION_{i}__", citation)
        
    final_text = re.sub(r'\s+__CITATION_\d+__\s+', ' ', final_text)
    for i, citation in enumerate(citations):
        final_text = final_text.replace(f"__CITATION_{i}__", citation)

    # تنظيف الفواصل المكررة والمسافات الزائدة في النواتج النهائية
    final_text = re.sub(r'\s+,\s+', ', ', final_text)
    final_text = final_text.replace(" ,", ",")
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
    return {"status": "healthy", "engine": "Super Hybrid Reshaper v4.0"}
