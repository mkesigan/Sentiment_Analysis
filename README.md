# 🌐 Twitter Sentiment Analysis (DistilBERT + FastAPI + React)

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![Tailwind](https://img.shields.io/badge/TailwindCSS-3.4-38BDF8?style=for-the-badge&logo=tailwindcss)
![HuggingFace](https://img.shields.io/badge/Transformers-DistilBERT-FFD21E?style=for-the-badge&logo=huggingface)


This is a complete **industry-grade NLP project**, built using modern machine learning, production engineering, and deployment practices. It is portfolio-ready and designed to demonstrate full-stack AI engineering skills.

### 🚀 Key Features
*   ✔ **DistilBERT Fine-tuning:** Customized transformer model for sentiment classification.
*   ✔ **Full ML Pipeline:** EDA → Preprocess → Train → Evaluate → Predict → Deploy.
*   ✔ **FastAPI Backend:** High-performance asynchronous inference API.
*   ✔ **Dockerized:** Containerized for consistent deployment anywhere.
*   ✔ **Modular Code:** Clean, structured, and maintainable codebase.

---

## 📁 Project Structure

```text
sentiment-analysis/
│── api/                      # FastAPI backend
│   ├── main.py
│   └── predict.py
│
│── sentiment-ui/             # React frontend
│   ├── src/
│   │   ├── App.js
│   │   └── index.css
│   └── package.json
│
│── src/                      # ML pipeline
│   ├── train_model.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   └── predict.py
│
│── models/
│   └── distilbert/
│
│── requirements.txt
│── README.md

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

8. React.js UI
An interactive dashboard for testing the model.
Run UI:
```Bash
cd sentiment-ui
npm install
npm start
# Runs at http://localhost:3000
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
### API Documentation
```Bash
POST /predict
{
  "text": "I love this!"
}

Response:
{
  "text": "I love this!",
  "clean_text": "i love this",
  "sentiment": "positive",
  "confidence": 0.97
}

```
### 📊 Dataset
Sentiment140 Dataset (1.6M Tweets)
* Labels: 0 (Negative), 4 (Positive)
* Source: Kaggle

### 🧠 Skills Demonstrated
**Category	           Skills**
**Machine Learning**   Text Cleaning, Tokenization, Transformer Fine-tuning, Evaluation Metrics,DistilBERT fine-tuning, tokenization
**Backend**            FastAPI, REST API, CORS
**DevOps**	           Docker, Containerization, Environment Isolation
**Frontend**	       React.js, Tailwind CSS, Chart.js visualization
**Engineering**	       Modular Code Structure, Git Version Control, Clean Documentation


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
