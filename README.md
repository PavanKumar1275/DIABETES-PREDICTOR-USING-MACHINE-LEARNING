# 🩺 Diabetes Prediction System

An advanced **Machine Learning-based Diabetes Prediction System** built using **Python**, **Streamlit**, and **Scikit-learn**.

This application predicts the likelihood of diabetes using medical parameters from the **PIMA Indians Diabetes Dataset**. It also provides interactive data visualization, multiple machine learning models, feature engineering, and secure user authentication.

---

## 🚀 Features

### 🔐 User Authentication
- User Registration
- Login using Email or Phone Number
- Secure Password Hashing
- SQLite Database

---

### 📊 Dataset Analysis
- Dataset Overview
- Statistical Summary
- Correlation Heatmap
- Feature Distribution
- Boxplots
- Class Distribution

---

### ⚙ Feature Engineering
- BMI Categories
- Age Groups
- Glucose Categories
- Interaction Features
- Polynomial Features
- Ratio Features
- Log Transformations
- Composite Health Score

---

### 🤖 Machine Learning Models

The system trains multiple models:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Voting Classifier (Ensemble)
- Stacking Classifier

---

### 📈 Model Evaluation

The application evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Cross Validation
- Confusion Matrix
- ROC Curve

---

### 🔮 Diabetes Prediction

Users can enter medical details and receive:

- Diabetes Prediction
- Prediction Probability
- Risk Assessment
- Recommended Model Results

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SQLite

---

## 📂 Project Structure

```
Diabetes-Prediction-System/
│
├── diabetes_prediction_app.py
├── diabetes_prediction_app_simple.py
├── user_auth.py
├── users.db
├── requirements.txt
├── run_app.bat
├── run_app.ps1
├── run_simple_app.bat
├── data/
│   ├── pima-indians-diabetes.csv
│   └── pima-indians-diabetes-processed.csv
│
└── README.md
```

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Diabetes-Prediction-System.git
```

```bash
cd Diabetes-Prediction-System
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Run Application

Using Streamlit

```bash
streamlit run diabetes_prediction_app.py
```

or

```bash
streamlit run diabetes_prediction_app_simple.py
```

Windows users can also run

```
run_app.bat
```

or

```
run_simple_app.bat
```

---

## 📋 Input Parameters

The prediction model uses:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

## 📊 Dataset

The project uses the famous

**PIMA Indians Diabetes Dataset**

containing medical diagnostic measurements for diabetes prediction.

---

## 🔒 Authentication

The application provides:

- User Registration
- Secure Login
- Password Hashing (SHA-256)
- SQLite Database Storage

---

## 📈 Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Feature Scaling
5. Feature Selection
6. Train/Test Split
7. Model Training
8. Hyperparameter Tuning
9. Model Evaluation
10. Diabetes Prediction

---

## 📸 Application Modules

- Login & Registration
- Dataset Overview
- Feature Engineering
- Model Training
- Model Evaluation
- Prediction Dashboard

---

## 📚 Future Improvements

- Deep Learning Models
- Explainable AI (SHAP)
- Cloud Deployment
- Email Notifications
- Medical Report Generation
- Patient History Tracking
- REST API
- Docker Support

---

## 👨‍💻 Author

**Pavan Kumar**

AI & Data Science Engineering Student

---

## ⭐ Support

If you found this project useful, please ⭐ star the repository.

Contributions, suggestions, and improvements are always welcome.
