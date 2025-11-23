import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
#from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# 1. Custom Dataset Class

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)



# 2. Compute evaluation metrics

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# 3. Main Training Function

def train_model():

    print("\n📌 Loading datasets...")
    train_df = pd.read_csv("../data/processed/train.csv")
    val_df   = pd.read_csv("../data/processed/val.csv")

    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_df = train_df.sample(n=50000, random_state=42).reset_index(drop=True)
    val_df   = val_df.sample(n=5000, random_state=42).reset_index(drop=True)


    print("📌 Creating dataset objects...")
    train_dataset = TweetDataset(train_df["clean_text"], train_df["label"], tokenizer)
    val_dataset   = TweetDataset(val_df["clean_text"], val_df["label"], tokenizer)

    print("📌 Loading BERT model...")
    #model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="../models/distilbert/",
        num_train_epochs=3,                 
        per_device_train_batch_size=16,     
        per_device_eval_batch_size=16,
        logging_steps=100,
        learning_rate=5e-5,                 # DistilBERT likes slightly higher LR
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        use_cpu=True,                       # Explicitly tell it to use CPU
        fp16=False                          # CPU doesn't support fp16 well
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("\n Training BERT model...")
    trainer.train()

    print("\n Saving final model...")
    model.save_pretrained("../models/distilbert/")
    tokenizer.save_pretrained("../models/distilbert/")

    print("\n Training complete! Model saved at /models/distilbert/")


if __name__ == "__main__":
    train_model()
