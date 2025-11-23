import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#Simple dataset for inference
class InferDataset(Dataset):
    def __init__(self,texts,tokenizer,max_len=128):
        self.texts=list(texts)
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text=str(self.texts[idx])
        encoding=self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors='pt'
        )

        return{
            "input_ids":encoding["input_ids"].squeeze(0),
            "attention_mask":encoding["attention_mask"].squeeze(0)

        }
    
#Batch inference helper
def run_inference(model,texts,tokenizer,batch_size=32,max_len=128,device='cpu'):
    dataset=InferDataset(texts,tokenizer,max_len=max_len)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False)

    model.to(device)
    model.eval()

    all_probs=[]

    with torch.no_grad():
        for batch in dataloader:
            input_ids=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            outputs=model(input_ids=input_ids,attention_mask=attention_mask)
            logits=outputs.logits
            probs=torch.softmax(logits,dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs=np.vstack(all_probs)
    return all_probs

#Main evaluation flow
def evaluate_model(model_dir="../models/distilbert/",test_csv="../data/processed/test.csv",batch_size=32,max_len=128):
    print("Loading test csv : ",test_csv)
    df_test=pd.read_csv(test_csv)
    texts=df_test["clean_text"].astype(str).tolist()
    true_labels=df_test["label"].astype(int).tolist()

    print("Loading tokenizer & model from ",model_dir)
    tokenizer=DistilBertTokenizer.from_pretrained(model_dir)
    model=DistilBertForSequenceClassification.from_pretrained(model_dir)

    #run inference(cpu)
    device='cpu'
    print("Running inference on device : ",device)
    probs=run_inference(model,texts,tokenizer,batch_size=batch_size,max_len=max_len,device=device)

    #prdicted labels and confidences
    pred_labels=probs.argmax(-1)
    pred_probs=probs.max(axis=1)

    # metrics
    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="binary")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["negative","positive"]))

    # confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("Confusion matrix:\n", cm)

    #save predictions to csv
    out_df=df_test.copy()
    out_df["pred_labels"]=pred_labels
    out_df["probs"]=pred_probs
    out_csv=os.path.join(os.path.dirname(test_csv),"test_predictions.csv")
    out_df.to_csv(out_csv)
    print("Saved predictions to : ",out_csv)


    #plot and confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=["neg","pos"],yticklabels=["neg","pos"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path=os.path.join(model_dir,"confusion_matrix.png")
    plt.savefig(cm_path,bbox_inches="tight")
    plt.show()
    plt.close()
    print("Save confusion matrix to :",cm_path)

    return{
        "accuracy":acc,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "pred_csv":out_csv,
        "confusion_matrix_png":cm_path
    }

if __name__=="__main__":
    evaluate_model()
 
