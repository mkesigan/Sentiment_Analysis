# 🌐 Twitter Sentiment Analysis (DistilBERT + FastAPI + Docker)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

This is a complete **industry-grade NLP project**, built using modern machine learning, production engineering, and deployment practices. It is portfolio-ready and designed to demonstrate full-stack AI engineering skills.

### 🚀 Key Features
*   ✔ **DistilBERT Fine-tuning:** Customized transformer model for sentiment classification.
*   ✔ **Full ML Pipeline:** EDA → Preprocess → Train → Evaluate → Predict → Deploy.
*   ✔ **FastAPI Backend:** High-performance asynchronous inference API.
*   ✔ **Dockerized:** Containerized for consistent deployment anywhere.
*   ✔ **Streamlit UI:** Interactive frontend for testing and visualization.
*   ✔ **Modular Code:** Clean, structured, and maintainable codebase.

---

## 📁 Project Structure

```text
sentiment-bert-project/
│── api/
│   └── main.py                    # FastAPI service
│
│── src/
│   ├── data_preprocessing.py      # Clean + split dataset
│   ├── train_model.py             # DistilBERT fine-tuning
│   ├── evaluate_model.py          # Metrics + confusion matrix
│   ├── predict.py                 # Inference logic used by API/UI
│   ├── ensemble.py                # (Optional) Voting / Stacking ensemble
│   └── utils.py                   # Helper functions
│
│── app/
│   └── streamlit_app.py           # Optional Streamlit frontend UI
│
│── models/
│   └── distilbert/                # Saved fine-tuned model + tokenizer
│
│── data/
│   ├── sentiment140.csv           # Raw dataset (Git ignored)
│   └── processed/                 # train.csv / val.csv / test.csv
│
│── Dockerfile                     # Production-ready container image
│── requirements.txt               # Python dependencies
│── README.md                      # Project documentation

```

### 🔄 Full Machine Learning Workflow
1. Exploratory Data Analysis (EDA)
Notebook: notebooks/01_EDA_sentiment140.ipynb
* Analysis: Dataset overview, Null/duplicate detection, Class distribution, Tweet length analysis, Word clouds.
* Note: EDA is only for analysis — no cleaning happens in this step.

2. Preprocessing Pipeline
Cleans text (removes URLs, mentions, emojis), handles lowercasing, and performs a stratified Train/Val/Test split.
```Bash
python src/data_preprocessing.py
```
Saves to: data/processed/

3. Train the DistilBERT Model
Fine-tunes a DistilBERT model using the HuggingFace Trainer API with validation per epoch.
```Bash
python src/train_model.py
```
Saves model to: models/distilbert/

4. Model Evaluation
Generates accuracy metrics, precision/recall/F1 scores, and a confusion matrix.
python src/evaluate_model.py

5. Ensemble Learning (Optional)
Supports Hard voting, Soft voting, and Stacking to boost predictive performance.
```Bash
python src/ensemble.py
```

6. Inference Script
The core engine for prediction. It cleans text, tokenizes, and returns sentiment + confidence.
```Bash
python src/predict.py
```

### 🛠️ API & Frontend
7. FastAPI Backend
The production-ready backend API.
Run locally:
```Bash
cd api
uvicorn main:app --reload
```
* Access API: http://127.0.0.1:8000
* Swagger Docs: http://127.0.0.1:8000/docs
  
Endpoints:
* GET /: Health check.
* POST /predict: Predict sentiment for input text.

8. Streamlit UI
An interactive dashboard for testing the model.
Run UI:
```Bash
streamlit run app/streamlit_app.py
```
*Features: Real-time prediction, Confidence visualization, Wordclouds, Batch CSV processing.

### 📦 Docker Containerization
To run the entire application in an isolated container:
1. Build the image:
```Bash
docker build -t sentiment-api .
```
2. Run the container:
```Bash
docker run -p 8000:8000 sentiment-api
```
3. Access:
```Bash
Go to http://localhost:8000/docs
```

### ☁️ Deployment Options
1.Google Cloud Run (Recommended): Serverless, auto-scaling, direct Docker support.
2.Render / Railway: Easiest for quick demos.
3.AWS EC2 / ECS: Full control for enterprise scaling.

### 📊 Dataset
Sentiment140 Dataset (1.6M Tweets)
* Labels: 0 (Negative), 4 (Positive)
* Source: Kaggle

### 🧠 Skills Demonstrated
**Category	          Skills**
**Machine Learning**	Text Cleaning, Tokenization, Transformer Fine-tuning, Evaluation Metrics, Ensemble Learning
**Backend**           Dev	FastAPI, Async Endpoints, Model Serving
**DevOps**	          Docker, Containerization, Environment Isolation
**Frontend**	        Streamlit, Data Visualization
**Engineering**	      Modular Code Structure, Git Version Control, Clean Documentation

### 🚀 Quick Start
1.Clone the repository:
```Bash
git clone https://github.com/your-username/sentiment-bert-project.git
cd sentiment-bert-project
```
2.Create virtual environment:
```Bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
3.Install dependencies:
```Bash
pip install -r requirements.txt
```
4.Run the pipeline:
```Bash
python src/data_preprocessing.py
python src/train_model.py
```
5.Start the API:
```Bashdocker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```
## 🚀 Live Demo

🔗 **Try the App Here:**  
https://sentiment5analysis.streamlit.app/


**License**: This project uses the publicly available Sentiment140 dataset.
