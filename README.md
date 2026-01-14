📧 Production-Grade Email Intelligence & Spam Classification Service
🔍 Overview
This project implements a production-grade Email Spam Classification service using classical machine learning models and a cloud-native deployment workflow.
The system is designed to classify incoming email text as SPAM or NOT SPAM, with precision-focused model selection to minimize false positives — a critical requirement for business email filtering systems.
The trained model is exposed via a FastAPI-based REST API, containerized using Docker, and successfully deployed on AWS EC2 for public inference access.
________________________________________
🧠 Architecture Overview
High-Level Flow
Client Request (Email Text)
        ↓
FastAPI (/predict)
        ↓
Text Preprocessing (NLTK)
        ↓
TF-IDF Vectorization
        ↓
Trained ML Model (Naive Bayes / Logistic Regression / XGBoost)
        ↓
Prediction Response (JSON)
________________________________________
🛠 Tech Stack
Core Technologies
•	Python 3.10
•	Scikit-learn – Classical ML models
•	NLTK – Text preprocessing
•	FastAPI – Inference API
•	Pydantic – Request/response validation
•	Uvicorn – ASGI server
Machine Learning
•	TF-IDF Vectorization
•	Multinomial Naive Bayes
•	Logistic Regression
•	XGBoost
•	Precision-oriented model evaluation
Deployment & DevOps
•	Docker – Containerized inference service
•	Docker Hub – Image registry
•	AWS EC2 (t2.micro) – Cloud deployment
•	Linux (Amazon Linux)
________________________________________
📂 Project Structure
Email-Spam-Classification/
│
├── main.py                # FastAPI inference service
├── model.pkl              # Trained ML model
├── TFIDF.pkl              # TF-IDF vectorizer
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── README.md              # Project documentation
└── notebooks/             # Model training & experimentation
________________________________________
🧠 Features Implemented
✅ Model Development & Evaluation
•	Benchmarked multiple classifiers:
o	Naive Bayes
o	Logistic Regression
o	Random Forest
o	XGBoost
o	Ensemble methods
•	Selected models based on precision–recall trade-offs
•	Optimized for low false-positive rate
✅ Text Processing Pipeline
•	Lowercasing
•	Tokenization (NLTK)
•	Stopword removal
•	Lemmatization
•	TF-IDF feature extraction
✅ Backend API
Endpoint
POST /predict
Request
{
  "text_input": "Congratulations! You have won a free voucher..."
}
Response
{
  "prediction": 1,
  "label": "Mail is SPAM"
}
✅ Dockerization
•	Fully containerized FastAPI service
•	Lightweight python:3.10-slim base image
•	Portable across cloud platforms
✅ AWS Deployment
•	Deployed on AWS EC2 (t2.micro)
•	Public API accessible via EC2 public IP
•	Verified inference via Swagger UI
•	Instances terminated post-validation to avoid costs
________________________________________
🔗 API Documentation (Swagger)
http://<EC2-PUBLIC-IP>:8000/docs
The Swagger UI allows:
•	Live request testing
•	Schema validation
•	Example payloads
________________________________________
⚙️ Local Setup
1️⃣ Clone Repository
git clone https://github.com/amitksingh2103/email-spam-detection.git
cd email-spam-detection
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run API
uvicorn main:app --host 0.0.0.0 --port 8000
5️⃣ Open Swagger UI
http://localhost:8000/docs
________________________________________
🐳 Docker Workflow
Build Image
docker build -t amitksingh2103/email-spam-detection:latest .
Push to Docker Hub
docker push amitksingh2103/email-spam-detection:latest
Run Container
docker run -p 8000:8000 amitksingh2103/email-spam-detection:latest
________________________________________
🧠 Design Decisions
•	TF-IDF over Word2Vec for interpretability and classical ML compatibility
•	Precision-first evaluation to reduce false positives
•	FastAPI chosen for speed, validation, and clean OpenAPI docs
•	Docker used to ensure reproducibility and portability
•	AWS EC2 selected for realistic cloud deployment demonstration
________________________________________
⚠️ Limitations
•	No authentication or rate limiting
•	No real-time monitoring/log aggregation
•	Model retraining pipeline not automated
•	Single-instance deployment (no autoscaling)
________________________________________
🚀 Future Improvements
•	Threshold tuning for dynamic precision/recall control
•	Add Prometheus + Grafana monitoring
•	Deploy on AWS ECS / EKS
•	Add authentication & request throttling
•	CI/CD pipeline with GitHub Actions
•	UI frontend (Streamlit / React)
________________________________________
👤 Author
Amit Kumar Singh
AI Engineer | Generative AI | Machine Learning | FastAPI | Docker | AWS

