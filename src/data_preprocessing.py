import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re


#Cleaning Function
def clean_text(text):
    """Cleans raw tweet text for model training."""
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    
    # Remove hashtag symbol
    text = re.sub(r"#", "", text)
    
    # Remove emojis safely
    text = text.encode("ascii", "ignore").decode("ascii")
    
    # Remove punctuation and special chars
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    
    # Convert multiple spaces -> single
    text = re.sub(r"\s+", " ", text)
    
    return text.lower().strip()

#Main preprocessing function
def preprocess_and_split(input_path="../data/sentiment140.csv",output_dir="../data/processed/"):

    print("\n📌Dataset Loading.....")

    df=pd.read_csv(input_path,encoding='latin-1',header=None)
    df.columns=["target","id","date","flag","user","text"]

    #keep relevant columns
    df=df[["target","text"]]

    print("📌Converting sentiment labels to numeric.....")
    df['label']=df["target"].map({0: 0,4: 1})
    df.drop(columns=["target"],inplace=True)

    print("📌Cleaning text.....")
    df["clean_text"]=df["text"].apply(clean_text)

    print("📌Dropping Nas and Duplicates.....")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df=df[["clean_text","label"]]

    #Train,Test and Validation Split
    train,temp=train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
        )
    
    val,test=train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        stratify=temp["label"]
    )


    print(f"Train Shape : {train.shape}")
    print(f"Test Shape : {test.shape}")
    print(f"Validation Shape : {val.shape}")

    #ensure outputdir exists
    os.makedirs(output_dir,exist_ok=True)

    print("📌Saving Files.....")
    train.to_csv(output_dir+"train.csv",index=False)
    val.to_csv(output_dir+"val.csv",index=False)
    test.to_csv(output_dir+"test.csv",index=False)

    print("\n The files are saved inside the /data/preprocessed/")


if __name__ == "__main__":
    preprocess_and_split()






    