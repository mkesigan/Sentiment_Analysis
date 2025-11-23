import torch
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification
import os
import re


#Text Cleaning

def clean_text(text):
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

#Load model & tokenizer

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "distilbert")

print(f"📌 Loading model from: {MODEL_PATH}")

tokenizer=DistilBertTokenizer.from_pretrained(MODEL_PATH)
model=DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

device='cpu'
model.to(device)
model.eval()

#Prediction function
def predict_sentiment(text :str):

    cleaned_text=clean_text(text)


    # Tokenize input
    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    probs = probs.cpu().numpy().flatten()
    pred_label = int(probs.argmax())
    confidence = float(probs.max())

    sentiment = "positive" if pred_label == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "clean_text": cleaned_text
    }

#Batch prediction

def predict_batch(text_list):

    results=[]

    for text in text_list:
        results.append(predict_sentiment(text))

    return results

#Debug
if __name__ =="__main__":
    sample="I Love Data Science !!! #machine learning"
    result=predict_sentiment(sample)
    print(result)