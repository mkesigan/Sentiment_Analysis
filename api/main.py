from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


#import predict.py file from src folder
current_dir=os.path.dirname(os.path.abspath(__file__))
src_path=os.path.join(current_dir,"..","src")
sys.path.append(src_path)

from predict import predict_sentiment

#define request body using pydantic
class TextInput(BaseModel):
    text:str

#initialize the fastapi app

app=FastAPI(
    title="Sentiment Analysis API",
    description="DistilBert model API for predicting sentiment(positive,negative)",
    version="1.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#base endpoint
@app.get("/")
def home():
    return {"message":"API is running! use /predict to classify sentiment."}

@app.post("/predict")
def predict(data:TextInput):
    result=predict_sentiment(data.text)
    return{
        "text":data.text,
        "clean_text":result["clean_text"],
        "sentiment":result["sentiment"],
        "confidence":result["confidence"]
    }

#run API
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
    