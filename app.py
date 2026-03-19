
import streamlit as st
import joblib, re, numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for p in ['stopwords','wordnet','omw-1.4']:
    nltk.download(p, quiet=True)

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="wide")

st.markdown('''
<style>
.title  { text-align:center; font-size:2.4rem; font-weight:800; color:#1e293b; }
.sub    { text-align:center; color:#64748b; font-size:1rem; margin-bottom:1rem; }
.pos-box{ background:#dcfce7; border:2px solid #16a34a; border-radius:12px;
          padding:1.2rem; text-align:center; font-size:1.5rem;
          font-weight:800; color:#15803d; margin:0.5rem 0; }
.neg-box{ background:#fee2e2; border:2px solid #dc2626; border-radius:12px;
          padding:1.2rem; text-align:center; font-size:1.5rem;
          font-weight:800; color:#b91c1c; margin:0.5rem 0; }
.stat   { background:#f1f5f9; border-radius:10px; padding:1rem;
          text-align:center; border:1px solid #e2e8f0; }
.snum   { font-size:1.6rem; font-weight:800; color:#1e293b; }
.slbl   { font-size:0.75rem; color:#64748b; text-transform:uppercase; }
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_all():
    model = joblib.load("sentiment_model.pkl")
    vec   = joblib.load("tfidf_vectorizer.pkl")
    lem   = WordNetLemmatizer()
    sw    = set(stopwords.words("english")) - {"no","not","never","nor","don't","doesn't"}
    return model, vec, lem, sw

model, vectorizer, lem, sw = load_all()

def clean(text):
    text = re.sub(r"<[^>]+>"," ", str(text).lower())
    text = re.sub(r"[^a-z\s]"," ", text)
    return " ".join([lem.lemmatize(w) for w in text.split()
                     if w not in sw and len(w) > 2])

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 About")
    st.markdown("---")
    st.markdown("**Dataset:** IMDB 50K Reviews")
    st.markdown("**Task:** Binary Sentiment Classification")
    st.markdown("**Features:** TF-IDF (unigrams + bigrams)")
    st.markdown("---")
    st.markdown("## 📋 Pipeline Stages")
    stages = [
        "1. Dataset Loading & EDA",
        "2. Text Preprocessing",
        "3. Feature Extraction",
        "4. Train Multiple Models",
        "5. Performance Evaluation",
        "6. Best Model Selection",
        "7. GUI Implementation"
    ]
    for s in stages:
        st.markdown(f"✅ {s}")
    st.markdown("---")
    st.markdown("## 🤖 Models Trained")
    for m in ["Logistic Regression","Naive Bayes",
              "Linear SVM","Decision Tree","Random Forest"]:
        st.markdown(f"• {m}")

# ── Main ─────────────────────────────────────────────────────
st.markdown("<p class='title'>🎬 Movie Review Sentiment Analyzer</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>AI-powered NLP system to detect Positive or Negative movie reviews</p>", unsafe_allow_html=True)
st.markdown("---")

col_input, col_result = st.columns([1.2, 1], gap="large")

with col_input:
    st.subheader("✍️ Enter Your Review")
    review = st.text_area("", height=200,
        placeholder="e.g. This movie was absolutely brilliant. The acting was superb...")

    predict_btn = st.button("🔍 Predict Sentiment", use_container_width=True)

    st.markdown("#### 💡 Try these examples:")
    examples = {
        "⭐ Positive": "This movie was absolutely fantastic! The acting was superb and the story was deeply moving.",
        "👎 Negative": "Terrible film. Complete waste of time. The plot made no sense and acting was awful.",
        "😐 Mixed":    "Not bad but not great either. Some good moments but mostly forgettable."
    }
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["example"] = text
            st.rerun()

    if "example" in st.session_state:
        review = st.session_state.pop("example")

with col_result:
    st.subheader("🎯 Prediction Result")

    if predict_btn and review.strip():
        cleaned   = clean(review)
        vec_input = vectorizer.transform([cleaned])
        pred      = model.predict(vec_input)[0]

        if pred == 1:
            st.markdown("<div class='pos-box'>😊 Positive Review</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='neg-box'>😡 Negative Review</div>",
                        unsafe_allow_html=True)

        try:
            score = model.decision_function(vec_input)[0]
            conf  = min(round(abs(score) * 12, 1), 99.9)
        except:
            conf  = round(max(model.predict_proba(vec_input)[0]) * 100, 1)

        st.markdown(f"**Confidence:** `{conf}%`")
        st.progress(min(conf / 100, 1.0))
        st.markdown("---")

        st.subheader("📊 Review Statistics")
        words = review.split()
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        c1.markdown(f"<div class='stat'><div class='snum'>{len(words)}</div><div class='slbl'>Total Words</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat'><div class='snum'>{len(review)}</div><div class='slbl'>Characters</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat'><div class='snum'>{len(cleaned.split())}</div><div class='slbl'>Tokens (processed)</div></div>", unsafe_allow_html=True)
        avg = round(np.mean([len(w) for w in words]), 1) if words else 0
        c4.markdown(f"<div class='stat'><div class='snum'>{avg}</div><div class='slbl'>Avg Word Length</div></div>", unsafe_allow_html=True)

        with st.expander("🔬 Preprocessed Text"):
            st.code(cleaned)

    elif predict_btn:
        st.warning("⚠️ Please enter a review first.")
    else:
        st.info("👈 Enter a review on the left and click **Predict Sentiment**")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#94a3b8;font-size:0.8rem;'>Movie Review Sentiment Analysis | NLP Project | Streamlit GUI</p>", unsafe_allow_html=True)
