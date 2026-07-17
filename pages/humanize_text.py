import random
import re
import ssl
import warnings
import nltk
import spacy
import streamlit as st
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore", category=FutureWarning)

########################################
# Download needed NLTK resources
########################################
def download_nltk_resources():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ['punkt', 'averaged_perceptron_tagger',
                 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng']
    for r in resources:
        nltk.download(r, quiet=True)

download_nltk_resources()

########################################
# Prepare spaCy pipeline
########################################
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy en_core_web_sm model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

########################################
# Citation Regex
########################################
CITATION_REGEX = re.compile(
    r"\(\s*[A-Za-z&\-,\.\s]+(?:et al\.\s*)?,\s*\d{4}(?:,\s*(?:pp?\.\s*\d+(?:-\d+)?))?\s*\)"
)

########################################
# Helper: Word & Sentence Counts
########################################
def count_words(text):
    return len(word_tokenize(text))

def count_sentences(text):
    return len(sent_tokenize(text))

########################################
# Step 1: Extract & Restore Citations
########################################
def extract_citations(text):
    refs = CITATION_REGEX.findall(text)
    placeholder_map = {}
    replaced_text = text
    for i, r in enumerate(refs, start=1):
        placeholder = f"[[REF_{i}]]"
        placeholder_map[placeholder] = r
        replaced_text = replaced_text.replace(r, placeholder, 1)
    return replaced_text, placeholder_map

PLACEHOLDER_REGEX = re.compile(r"\[\s*\[\s*REF_(\d+)\s*\]\s*\]")

def restore_citations(text, placeholder_map):
    def replace_placeholder(match):
        idx = match.group(1)
        key = f"[[REF_{idx}]]"
        return placeholder_map.get(key, match.group(0))

    restored = PLACEHOLDER_REGEX.sub(replace_placeholder, text)
    return restored

########################################
# Step 2: Expansions, Synonyms, & Transitions
########################################
WHOLE_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not", "ain't": "is not",
    "i'm": "i am", "it's": "it is", "we're": "we are", "they're": "they are",
    "you're": "you are", "he's": "he is", "she's": "she is", "that's": "that is",
    "there's": "there is", "what's": "what is", "who's": "who is", "let's": "let us",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not", "isn't": "is not",
    "aren't": "are not", "weren't": "were not", "hasn't": "has not", "haven't": "have not", "hadn't": "had not"
}

SUFFIX_CONTRACTIONS = {
    "n't": " not", "'re": " are", "'s": " is", "'ll": " will", "'ve": " have", "'d": " would", "'m": " am"
}

ACADEMIC_TRANSITIONS = [
    "Moreover,", "Additionally,", "Furthermore,", "Hence,", "Therefore,",
    "Consequently,", "Nonetheless,", "Nevertheles,", "In contrast,",
    "On the other hand,", "In addition,", "As a result,"
]

def expand_contractions(sentence):
    def _replace_whole_with_quotes(match):
        open_tok = match.group(1) or ""
        word = match.group('word')
        close_tok = match.group(3) or ""
        key = word.lower()
        repl = WHOLE_CONTRACTIONS.get(key, word)
        if word and word[0].isupper():
            repl = repl.capitalize()
        return f"{open_tok}{repl}{close_tok}"

    alt = "|".join(re.escape(k) for k in WHOLE_CONTRACTIONS.keys())
    whole_pattern = rf"(?:(``)\s*)?(?P<word>(?:{alt}))(?:\s*(''))?"
    sentence = re.sub(whole_pattern, _replace_whole_with_quotes, sentence, flags=re.IGNORECASE)

    tokens = word_tokenize(sentence)
    out_tokens = []
    for t in tokens:
        lower_t = t.lower()
        replaced = False
        for contr, expansion in SUFFIX_CONTRACTIONS.items():
            if lower_t.endswith(contr):
                base = lower_t[: -len(contr)]
                new_t = base + expansion
                if t and t[0].isupper():
                    new_t = new_t.capitalize()
                out_tokens.append(new_t)
                replaced = True
                break
        if not replaced:
            out_tokens.append(t)
    return " ".join(out_tokens)

def replace_synonyms(sentence, p_syn=0.2):
    if not nlp:
        return sentence

    doc = nlp(sentence)
    new_tokens = []
    
    for token in doc:
        # 1. حماية المراجع الإشارية كلياً
        if "[[REF_" in token.text:
            new_tokens.append(token.text + token.whitespace_)
            continue
            
        # 2. حماية أسماء الأعلام، العناوين، والكيانات المسمّاة (مثل Hamlet و Shakespeare)
        is_first_word = (token.i == 0)
        is_capitalized = token.text[0].isupper() if (token.text and token.text[0].isalpha()) else False
        
        if token.pos_ == "PROPN" or token.ent_type_ != "" or (is_capitalized and not is_first_word):
            new_tokens.append(token.text + token.whitespace_)
            continue
            
        # 3. استبدال ذكي وآمن للمفردات مع الحفاظ على الفراغات الأصلية ومنع الانحراف الدلالي
        if token.pos_ in ["ADJ", "NOUN", "VERB", "ADV"] and wordnet.synsets(token.text):
            if random.random() < p_syn:
                synonyms = get_synonyms(token.text, token.pos_)
                if synonyms:
                    chosen_syn = random.choice(synonyms)
                    # الحفاظ على حالة الحروف إذا كانت الكلمة في بداية الجملة
                    if is_capitalized:
                        chosen_syn = chosen_syn.capitalize()
                    new_tokens.append(chosen_syn + token.whitespace_)
                else:
                    new_tokens.append(token.text + token.whitespace_)
            else:
                new_tokens.append(token.text + token.whitespace_)
        else:
            new_tokens.append(token.text + token.whitespace_)
            
    return "".join(new_tokens)

def get_synonyms(word, pos):
    wn_pos = None
    if pos.startswith("ADJ"): wn_pos = wordnet.ADJ
    elif pos.startswith("NOUN"): wn_pos = wordnet.NOUN
    elif pos.startswith("ADV"): wn_pos = wordnet.ADV
    elif pos.startswith("VERB"): wn_pos = wordnet.VERB

    synonyms = set()
    if wn_pos:
        # تقييد دلالي لأول سياقين فقط لمنع الخلط بين المعاني المشتركة (مثل أعمال أدبية ومصانع)
        synsets = wordnet.synsets(word, pos=wn_pos)
        for syn in synsets[:2]: 
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace("_", " ")
                # حظر الكلمات المتطابقة والعبارات المركبة لتجنب ركاكة التركيب النحوي
                if lemma_name.lower() != word.lower() and " " not in lemma_name:
                    synonyms.add(lemma_name)
    return list(synonyms)

def add_academic_transition(sentence, p_transition=0.2):
    if random.random() < p_transition:
        transition = random.choice(ACADEMIC_TRANSITIONS)
        return f"{transition} {sentence}"
    return sentence

########################################
# Step 3: Minimal "Humanize" line-by-line
########################################
def minimal_humanize_line(line, p_syn=0.2, p_trans=0.2, use_passive=False):
    line = expand_contractions(line)
    line = replace_synonyms(line, p_syn=p_syn)
    line = add_academic_transition(line, p_transition=p_trans)
    return line

def minimal_rewriting(text, p_syn=0.2, p_trans=0.2, use_passive=False):
    lines = sent_tokenize(text)
    out_lines = [
        minimal_humanize_line(ln, p_syn=p_syn, p_trans=p_trans, use_passive=use_passive) for ln in lines
    ]
    return " ".join(out_lines)

def preserve_linebreaks_rewrite(text, p_syn=0.2, p_trans=0.2, use_passive=False):
    lines = text.splitlines()
    out_lines = []
    for ln in lines:
        if not ln.strip():
            out_lines.append("")
        else:
            out_lines.append(minimal_rewriting(
                ln, p_syn=p_syn, p_trans=p_trans, use_passive=use_passive))
    return "\n".join(out_lines)

########################################
# Final: Show Humanize Page
########################################
def show_humanize_page():
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Main", type="secondary"):
            st.session_state["current_page"] = "Main"
            st.rerun()
    with col2:
        if st.button("Switch to PDF Detection →", type="secondary"):
            st.session_state["current_page"] = "PDF Detection & Annotation"
            st.rerun()
    
    st.title("✍️ AI Text Humanizer & Enhancer")
    st.markdown("---")

    # تهيئة واجهة التحكم في السلايدرات
    st.subheader("🎛️ Customize Your Humanization Settings")
    col1, col2 = st.columns(2)
    with col1:
        p_syn = st.slider("**Synonym Replacement Intensity**", 0.0, 1.0, 0.2, 0.05)
    with col2:
        p_trans = st.slider("**Academic Transition Frequency**", 0.0, 1.0, 0.2, 0.05)

    # إضافة السويتش الخاص بالـ Passive Voice ليتزامن مع كود الباكيند والواجهات
    use_passive = st.checkbox("**Enable Structural Transformation (Passive Voice)**", value=False)

    st.subheader("📝 Enter Your Text to Humanize")
    input_text = st.text_area("Paste your AI-generated text below:", height=200, label_visibility="collapsed")

    if st.button("🚀 Humanize Text", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("📝 Please enter some text to humanize first.")
            return

        orig_wc = count_words(input_text)
        orig_sc = count_sentences(input_text)

        with st.spinner("✍️ Processing text with advanced linguistic rules..."):
            no_refs_text, placeholders = extract_citations(input_text)
            partially_rewritten = preserve_linebreaks_rewrite(
                no_refs_text, p_syn=p_syn, p_trans=p_trans, use_passive=use_passive
            )
            final_text = restore_citations(partially_rewritten, placeholders)

            # معالجة وتنظيف الفراغات الشاذة حول علامات الترقيم نتاج التفكيك
            final_text = re.sub(r"[ \t]+([.,;:!?])", r"\1", final_text)
            final_text = re.sub(r"(\()[ \t]+", r"\1", final_text)
            final_text = re.sub(r"[ \t]+(\))", r"\1", final_text)
            final_text = re.sub(r"[ \t]{2,}", " ", final_text)
            final_text = re.sub(r"``\s*(.+?)\s*''", r'"\1"', final_text)

        new_wc = count_words(final_text)
        new_sc = count_sentences(final_text)

        st.subheader("🎉 Your Humanized Text")
        st.text_area("Humanized Result", final_text, height=300, label_visibility="collapsed")

        st.download_button("📋 Download Humanized Text", data=final_text, file_name="humanized_text.txt", mime="text/plain", use_container_width=True)

if __name__ == "__main__":
    show_humanize_page()
