🌐 Twitter Sentiment Analysis using DistilBERT + FastAPI + Docker + Streamlit (End-to-End ML Project)

This is a complete industry-grade NLP project, built using modern machine learning, production engineering, and deployment practices.

It includes:

✔ DistilBERT fine-tuning
✔ Full ML pipeline (EDA → Preprocess → Train → Evaluate → Predict → Deploy)
✔ FastAPI inference backend
✔ Docker containerization
✔ Optional Streamlit UI
✔ Clean modular code structure
✔ Cloud deployable (Google Cloud Run / Render / Railway)

This project is portfolio-ready and ideal for AI/ML internship applications.

📁 Project Structure
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
│
│── app/
│   └── streamlit_app.py           # Optional Streamlit frontend UI
│
│── models/
│   └── distilbert/                # Saved fine-tuned model + tokenizer
│
│── data/
│   ├── sentiment140.csv           # Raw dataset
│   └── processed/                 # train.csv / val.csv / test.csv
│
│── Dockerfile                     # Production-ready container
│── requirements.txt               # Python dependencies
│── README.md                      # Project documentation

🔄 Full Machine Learning Workflow
1. Exploratory Data Analysis (EDA)

Notebook:
notebooks/01_EDA_sentiment140.ipynb

Includes:

Dataset overview

Null/duplicate detection

Class distribution

Tweet length analysis

Word clouds

💡 EDA is only for analysis — no cleaning in this step.

2. Preprocessing Pipeline (Production Cleaning)

Run:

python src/data_preprocessing.py


Tasks:

Remove URLs, mentions, emojis

Remove punctuation and unnecessary characters

Lowercasing

Stratified Train/Val/Test split

Saves to: data/processed/

3. Train the DistilBERT Model

Run:

python src/train_model.py


Includes:

Tokenization

Dataset creation

HuggingFace Trainer

Validation per epoch

Saves model to: models/distilbert/

4. Model Evaluation

Run:

python src/evaluate_model.py


Generates:

Accuracy

Precision, Recall, F1

Confusion matrix PNG

CSV with predictions

Misclassified samples

5. Ensemble Learning (Optional)

Run:

python src/ensemble.py


Supports:

Hard voting

Soft voting

Stacking

Boosts predictive performance.

6. Inference Script (Core Engine for API/UI)

Run:

python src/predict.py


Features:

Clean text

Tokenize

Predict sentiment + confidence

Supports single or batch prediction

Used directly by:

FastAPI backend

Streamlit UI

Docker container

Cloud deployment

7. FastAPI Backend (Production API)

Run locally:

cd api
python main.py


API URLs:

http://127.0.0.1:8000
http://127.0.0.1:8000/docs


Endpoints:

GET / → health check

POST /predict → predict sentiment

8. Streamlit UI (Optional Frontend)

Run:

streamlit run app/streamlit_app.py


Features:

Text input sentiment prediction

Confidence score visualization

Sentiment distribution

Wordcloud

CSV batch prediction

📦 Docker Containerization

Build image:

docker build -t sentiment-api .


Run container:

docker run -p 8000:8000 sentiment-api


Access API:

http://localhost:8000/docs

☁️ Deployment Options
✔ Google Cloud Run (Recommended)

Fully serverless

Auto-scaling

Free tier handles millions of requests

Direct Docker support

✔ Render / Railway

Very easy deployment

No cloud complexity

✔ AWS EC2 / ECS

More advanced

Full control over VM / containers

📊 Dataset

Sentiment140 Dataset (1.6M Tweets)
Labels:

0 → Negative

4 → Positive

Source: Kaggle (public)

🧠 Skills Demonstrated
Machine Learning & NLP

Tweet text cleaning

Tokenization

Transformer model fine-tuning

Model evaluation

Ensemble learning

Inference optimization

Backend Development

FastAPI

Model serving

Production endpoints

Deployment Engineering

Docker

Container orchestration

Cloud deployment (Cloud Run)

Frontend

Streamlit

UI layout and visualization

API communication

Software Engineering Best Practices

Modular project structure

Version control

Environment isolation

Clean documentation

🚀 How to Run the Entire Project
1. Clone the repository
git clone <repo-url>
cd sentiment-bert-project

2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run preprocessing, training, evaluation

(See workflow above)

5. Start FastAPI (through Docker)
docker run -p 8000:8000 sentiment-api

📜 License

This project uses the publicly available Sentiment140 dataset from Kaggle.