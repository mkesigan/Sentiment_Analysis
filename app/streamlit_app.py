import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import re
import pandas as pd


# Page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>
/* Center container */
.center-container {
    max-width: 800px;
    margin: auto;
}

/* Title styling */
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #222222;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #666;
    margin-top: -10px;
}

/* Card */
.card {
    background: #ffffff;
    padding: 20px 25px;
    border-radius: 12px;
    border: 1px solid #e5e5e5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-top: 25px;
}

/* Center button */
.stButton > button {
    width: 200px;
    margin: auto;
    display: block;
    border-radius: 6px;
    background-color: #4F46E5;
    color: white;
}
</style>
""", unsafe_allow_html=True)


#Model loading
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("../models/distilbert")
    tokenizer = DistilBertTokenizer.from_pretrained("../models/distilbert")
    return model, tokenizer

model, tokenizer = load_model()


#Clean function
def clean_text(text):
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


# Main ui
st.markdown("<div class='center-container'>", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>💬 Sentiment Analysis with DistilBERT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A clean and interactive machine learning interface</p>", unsafe_allow_html=True)

# Input Card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("### ✍️ Enter Text")
text_input = st.text_area("", height=150, placeholder="Type something here...")
analyze_btn = st.button("Analyze Sentiment")
st.markdown("</div>", unsafe_allow_html=True)


# Output
if analyze_btn and text_input.strip():
    clean = clean_text(text_input)

    inputs = tokenizer(clean, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).flatten()

    prob_neg, prob_pos = float(probs[0]), float(probs[1])
    sentiment = "Positive 😊" if prob_pos > prob_neg else "Negative 😠"
    confidence = max(prob_pos, prob_neg)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### 📌 Prediction Result")
    st.write(f"#### {sentiment}")
    st.write(f"**Confidence:** {confidence:.2%}")

    
    explanation = ""

    positive_words = ["good", "great", "love", "amazing", "happy", "excellent", "fantastic"]
    negative_words = ["bad", "terrible", "hate", "angry", "worst", "disappointed", "sad"]

    matched_pos = [w for w in positive_words if w in clean.split()]
    matched_neg = [w for w in negative_words if w in clean.split()]

    if sentiment.startswith("Positive"):
        if matched_pos:
            explanation = (
                f"The model predicted **Positive** because the text contains positive words like "
                f"{', '.join([f'\"{w}\"' for w in matched_pos])}. "
                f"These words typically express appreciation, satisfaction or happiness.\n\n"
                f"The model is **{confidence:.1%} confident**, which indicates a strong positive signal."
            )
        else:
            explanation = (
                "The model predicted **Positive** based on the general tone and structure of the sentence. "
                "Even though no strong positive keywords were found, BERT captures contextual meaning."
            )

    else:  # Negative case
        if matched_neg:
            explanation = (
                f"The model predicted **Negative** because the text contains negative words such as "
                f"{', '.join([f'\"{w}\"' for w in matched_neg])}. "
                f"These terms express anger, dissatisfaction or criticism.\n\n"
                f"The model is **{confidence:.1%} confident**, showing strong negative signal."
            )
        else:
            explanation = (
                "The model predicted **Negative** based on the overall tone. "
                "Although explicit negative words were not found, the model detected contextual negativity."
            )

    # Show explanation section
    st.write("### 🧠 Explanation")
    st.info(explanation)

    st.write("---")

    # Probability chart
    df = pd.DataFrame({
        "Sentiment": ["Negative", "Positive"],
        "Confidence": [prob_neg, prob_pos]
    })
    st.write("### 📊 Probability Chart")
    st.bar_chart(df.set_index("Sentiment"))

    # Clean text
    with st.expander("🔍 View Cleaned Text"):
        st.write(clean)

    st.markdown("</div>", unsafe_allow_html=True)
