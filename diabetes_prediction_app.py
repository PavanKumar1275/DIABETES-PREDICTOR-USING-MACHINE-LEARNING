import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import warnings
import os

# Import user authentication module
from user_auth import register_user, authenticate_user

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    background-attachment: fixed;
    color: #333;
}

.stApp {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    padding: 30px;
    margin: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

h1, h2, h3, h4, h5, h6 {
    font-weight: 700;
    color: #2c3e50;
    letter-spacing: -0.5px;
}

h1 {
    font-size: 2.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
    background: linear-gradient(90deg, #3498db, #2c3e50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 {
    font-size: 1.8rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
    position: relative;
    padding-bottom: 10px;
}

h2:after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 60px;
    height: 4px;
    background: linear-gradient(90deg, #3498db, #2c3e50);
    border-radius: 2px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, #3498db, #2c3e50);
    border-radius: 12px;
    padding: 8px;
    box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
}

.stTabs [data-baseweb="tab"] {
    color: rgba(255, 255, 255, 0.8);
    font-weight: 600;
    font-size: 1rem;
    padding: 12px 20px;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: white;
    background-color: rgba(255, 255, 255, 0.1);
}

.stTabs [aria-selected="true"] {
    background-color: white;
    color: #2c3e50;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #27ae60, #219653);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(39, 174, 96, 0.3);
    cursor: pointer;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #219653, #1e8449);
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(39, 174, 96, 0.4);
}

.stButton>button:active {
    transform: translateY(0);
}

/* Input styling */
.stTextInput>div>div>input,
.stSelectbox>div>div>select,
.stNumberInput>div>div>input {
    border-radius: 10px;
    border: 2px solid #e1e8ed;
    padding: 12px 15px;
    font-size: 1rem;
    transition: all 0.3s ease;
    background-color: #fafafa;
}

.stTextInput>div>div>input:focus,
.stSelectbox>div>div>select:focus,
.stNumberInput>div>div>input:focus {
    border-color: #3498db;
    box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
    background-color: white;
}

/* Data frame styling */
.stDataFrame {
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    overflow: hidden;
    border: 1px solid #e1e8ed;
}

.stDataFrame table {
    border-collapse: separate;
    border-spacing: 0;
}

.stDataFrame th {
    background: linear-gradient(90deg, #3498db, #2c3e50);
    color: white;
    font-weight: 600;
    padding: 15px;
    text-align: left;
}

.stDataFrame td {
    padding: 12px 15px;
    border-bottom: 1px solid #e1e8ed;
}

.stDataFrame tr:last-child td {
    border-bottom: none;
}

.stDataFrame tr:nth-child(even) {
    background-color: #f8f9fa;
}

.stDataFrame tr:hover {
    background-color: #e3f2fd;
    transition: background-color 0.2s ease;
}

/* Metric styling */
.stMetric {
    background: linear-gradient(135deg, #ffffff, #f8f9fa);
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    padding: 20px;
    text-align: center;
    border: 1px solid #e1e8ed;
    transition: transform 0.3s ease;
}

.stMetric:hover {
    transform: translateY(-5px);
}

.stMetric label {
    font-weight: 600;
    color: #7f8c8d;
    font-size: 1.1rem;
}

.stMetric div {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2c3e50;
    margin-top: 10px;
}

/* Alert styling */
.stAlert {
    border-radius: 12px;
    padding: 15px 20px;
    margin: 15px 0;
    border: none;
    font-weight: 500;
}

.stSuccess {
    background: linear-gradient(90deg, #d4edda, #c3e6cb);
    color: #155724;
    border-left: 5px solid #28a745;
}

.stWarning {
    background: linear-gradient(90deg, #fff3cd, #ffeaa7);
    color: #856404;
    border-left: 5px solid #ffc107;
}

.stError {
    background: linear-gradient(90deg, #f8d7da, #f5c6cb);
    color: #721c24;
    border-left: 5px solid #dc3545;
}

.stInfo {
    background: linear-gradient(90deg, #d1ecf1, #bee5eb);
    color: #0c5460;
    border-left: 5px solid #17a2b8;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    color: white;
    border-radius: 0 20px 20px 0;
    padding: 20px;
    box-shadow: 3px 0 15px rgba(0, 0, 0, 0.1);
}

.sidebar .sidebar-content h1, 
.sidebar .sidebar-content h2, 
.sidebar .sidebar-content h3 {
    color: white;
}

.sidebar .sidebar-content .stButton>button {
    background: linear-gradient(90deg, #3498db, #2980b9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 15px;
    font-weight: 600;
    width: 100%;
    margin: 5px 0;
}

.sidebar .sidebar-content .stButton>button:hover {
    background: linear-gradient(90deg, #2980b9, #2573a7);
    transform: translateY(-1px);
}

/* Animation keyframes */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Apply animations */
.stMarkdown, .stDataFrame, .stMetric, .stAlert {
    animation: fadeIn 0.5s ease-out;
}

/* Responsive design */
@media (max-width: 768px) {
    .stApp {
        padding: 15px;
        margin: 10px;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    h2 {
        font-size: 1.5rem;
    }
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(90deg, #3498db, #2c3e50);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(90deg, #2980b9, #1a2530);
}

/* Card styling */
.card {
    background: white;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    padding: 20px;
    margin: 15px 0;
    border: 1px solid #e1e8ed;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* Header styling */
.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    margin-bottom: 10px;
}

.header p {
    font-size: 1.2rem;
    color: #7f8c8d;
    max-width: 800px;
    margin: 0 auto;
}

/* Footer styling */
.footer {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e1e8ed;
    color: #7f8c8d;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# Check if user is logged in
if 'user' not in st.session_state:
    # Show login/signup page
    st.markdown("""
    <div class="header">
        <h1>🩺 Diabetes Prediction System</h1>
        <p>Advanced machine learning-powered diabetes risk assessment using the PIMA Indian Diabetes Database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for login and signup
    auth_tab1, auth_tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with auth_tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Login to Your Account")
        login_identifier = st.text_input("📧 Email or Phone Number", key="login_identifier")
        login_password = st.text_input("🔒 Password", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Login", use_container_width=True):
                if login_identifier and login_password:
                    success, result = authenticate_user(login_identifier, login_password)
                    if success:
                        st.session_state.user = result
                        st.success(f"Welcome back, {result['name']}!")
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.error("Please enter both email/phone and password")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with auth_tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Create New Account")
        signup_name = st.text_input("👤 Full Name", key="signup_name")
        signup_email = st.text_input("📧 Email Address", key="signup_email")
        signup_phone = st.text_input("📱 Phone Number", key="signup_phone")
        signup_password = st.text_input("🔒 Password", type="password", key="signup_password")
        signup_confirm_password = st.text_input("🔒 Confirm Password", type="password", key="signup_confirm_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Sign Up", use_container_width=True):
                if signup_password != signup_confirm_password:
                    st.error("Passwords do not match")
                elif signup_name and signup_email and signup_phone and signup_password:
                    success, message = register_user(signup_email, signup_phone, signup_password, signup_name)
                    if success:
                        st.success(message)
                        # Automatically log in the user
                        success, result = authenticate_user(signup_email, signup_password)
                        if success:
                            st.session_state.user = result
                            st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>About This Application</h2>
        <p>This advanced application predicts the likelihood of diabetes based on medical parameters using state-of-the-art machine learning algorithms.</p>
        <p>The system is based on the renowned PIMA Indian Diabetes Database and incorporates enhancements over the original IEEE paper, featuring:</p>
        <ul>
            <li>Five sophisticated ML models: Logistic Regression, Random Forest, SVM, KNN, and Voting Classifier</li>
            <li>Comprehensive feature engineering for improved accuracy</li>
            <li>Detailed model evaluation with multiple metrics</li>
            <li>Interactive data visualization and analysis</li>
        </ul>
        <p>To access the full functionality, please login or sign up.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <p>Diabetes Prediction System v2.0 | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # User is logged in, show the main application
    st.markdown(f"""
    <div class="header">
        <h1>🩺 Diabetes Prediction System</h1>
        <p>Welcome, {st.session_state.user['name']}! Advanced machine learning-powered diabetes risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add logout button in sidebar
    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<h3>👤 {st.session_state.user['name']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>📧 {st.session_state.user['email']}</p>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            del st.session_state.user
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>📊 Application Features</h3>
            <ul>
                <li>Dataset Overview & Analysis</li>
                <li>Advanced Feature Engineering</li>
                <li>Multiple ML Model Training</li>
                <li>Comprehensive Model Evaluation</li>
                <li>Real-time Diabetes Prediction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>This advanced application predicts the likelihood of diabetes based on medical parameters using state-of-the-art machine learning algorithms.</p>
        <p>The system is based on the renowned PIMA Indian Diabetes Database and incorporates enhancements over the original IEEE paper.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load and prepare data
    @st.cache_data(ttl=3600, max_entries=10)
    def load_data():
        # Load the actual PIMA Indian Diabetes dataset
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
            
            # Try to load the processed dataset first
            processed_file = os.path.join(data_dir, 'pima-indians-diabetes-processed.csv')
            if os.path.exists(processed_file):
                df = pd.read_csv(processed_file)
            else:
                # If processed dataset doesn't exist, load raw dataset and add column names
                raw_file = os.path.join(data_dir, 'pima-indians-diabetes.csv')
                column_names = [
                    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
                ]
                df = pd.read_csv(raw_file, names=column_names)
                # Save the processed dataset
                df.to_csv(processed_file, index=False)
            return df
        except Exception as e:
            st.warning(f"Could not load the dataset: {e}. Using synthetic data instead.")
            # Create synthetic PIMA Indian Diabetes dataset as fallback
            np.random.seed(42)
            
            # Generate synthetic data similar to PIMA Indian dataset (reduced for performance)
            n_samples = 200
            
            data = {
                'Pregnancies': np.random.randint(0, 18, n_samples),
                'Glucose': np.random.randint(0, 200, n_samples),
                'BloodPressure': np.random.randint(0, 123, n_samples),
                'SkinThickness': np.random.randint(0, 99, n_samples),
                'Insulin': np.random.randint(0, 846, n_samples),
                'BMI': np.random.uniform(0, 67, n_samples),
                'DiabetesPedigreeFunction': np.random.uniform(0, 2.42, n_samples),
                'Age': np.random.randint(21, 81, n_samples)
            }
            
            # Create target variable with some correlation to features
            diabetes_prob = (
                (data['Glucose'] > 140) * 0.3 +
                (data['BMI'] > 30) * 0.2 +
                (data['Age'] > 50) * 0.15 +
                (data['Pregnancies'] > 5) * 0.1 +
                np.random.random(n_samples) * 0.25
            )
            
            data['Outcome'] = (diabetes_prob > 0.5).astype(int)
            
            df = pd.DataFrame(data)
            return df

    # Enhanced feature engineering
    @st.cache_data(ttl=3600, max_entries=20)
    def engineer_features(df):
        df = df.copy()
        
        # Derived features
        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
        df['Glucose_Category'] = pd.cut(df['Glucose'], bins=[0, 100, 140, 200], labels=['Normal', 'Prediabetic', 'Diabetic'])
        
        # Interaction features
        df['BMI_Age'] = df['BMI'] * df['Age']
        df['Glucose_BMI'] = df['Glucose'] * df['BMI']
        df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
        
        # Polynomial features
        df['Glucose_Squared'] = df['Glucose'] ** 2
        df['BMI_Squared'] = df['BMI'] ** 2
        df['Age_Squared'] = df['Age'] ** 2
        
        # Ratio features
        df['BMI_Ratio'] = df['BMI'] / (df['Age'] + 1)  # Add 1 to avoid division by zero
        df['Glucose_Ratio'] = df['Glucose'] / (df['BMI'] + 1)  # Add 1 to avoid division by zero
        
        # Log transformations for skewed features
        df['Log_Insulin'] = np.log(df['Insulin'] + 1)  # Add 1 to handle zero values
        df['Log_DiabetesPedigreeFunction'] = np.log(df['DiabetesPedigreeFunction'] + 1)  # Add 1 to handle zero values
        
        # Binning for numerical features
        df['Pregnancies_Bin'] = pd.cut(df['Pregnancies'], bins=[-1, 0, 2, 5, 17], labels=['None', 'Low', 'Medium', 'High'])
        df['SkinThickness_Bin'] = pd.cut(df['SkinThickness'], bins=[0, 20, 30, 50, 100], labels=['Low', 'Normal', 'High', 'VeryHigh'])
        
        # Health indicators
        df['Glucose_BloodPressure_Ratio'] = df['Glucose'] / (df['BloodPressure'] + 1)  # Add 1 to avoid division by zero
        df['BMI_Age_Ratio'] = df['BMI'] / (df['Age'] + 1)  # Add 1 to avoid division by zero
        
        # Composite health score (simplified)
        df['Health_Score'] = (
            (df['Glucose'] / 200) * 0.3 +
            (df['BMI'] / 50) * 0.2 +
            (df['Age'] / 100) * 0.2 +
            (df['Pregnancies'] / 20) * 0.1 +
            (df['BloodPressure'] / 150) * 0.2
        )
        
        return df

    # Prepare data for modeling
    @st.cache_data(ttl=3600, max_entries=20)
    def prepare_data(df):
        # Handle missing values (zeros in the dataset)
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                    'DiabetesPedigreeFunction', 'Age']
        
        # Replace 0s with NaN for certain features
        df_processed = df.copy()
        df_processed[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_processed[
            ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
        
        # Fill missing values with median
        for feature in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df_processed[feature].fillna(df_processed[feature].median(), inplace=True)
        
        return df_processed

    # Train models
    @st.cache_resource(ttl=3600, max_entries=10)
    def train_models(X_train, y_train, hyperparameter_tuning=False):
        models = {}
        
        if hyperparameter_tuning:
            # Logistic Regression with comprehensive hyperparameter tuning
            lr_params = {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
            lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), 
                              lr_params, cv=3, scoring='accuracy', n_jobs=-1)
            lr.fit(X_train, y_train)
            models['Logistic Regression'] = lr.best_estimator_
            
            # Random Forest with comprehensive hyperparameter tuning
            rf_params = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            rf = GridSearchCV(RandomForestClassifier(random_state=42), 
                              rf_params, cv=3, scoring='accuracy', n_jobs=-1)
            rf.fit(X_train, y_train)
            models['Random Forest'] = rf.best_estimator_
            
            # SVM with comprehensive hyperparameter tuning
            svm_params = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
            svm = GridSearchCV(SVC(probability=True, random_state=42), 
                               svm_params, cv=3, scoring='accuracy', n_jobs=-1)
            svm.fit(X_train, y_train)
            models['SVM'] = svm.best_estimator_
            
            # KNN with comprehensive hyperparameter tuning
            knn_params = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            knn = GridSearchCV(KNeighborsClassifier(), 
                               knn_params, cv=3, scoring='accuracy', n_jobs=-1)
            knn.fit(X_train, y_train)
            models['KNN'] = knn.best_estimator_
        else:
            # Basic models without hyperparameter tuning - optimized for performance
            models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
            models['Random Forest'] = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=5)
            models['SVM'] = SVC(probability=True, random_state=42, kernel='rbf')
            models['KNN'] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            
            # Fit all models
            models['Logistic Regression'].fit(X_train, y_train)
            models['Random Forest'].fit(X_train, y_train)
            models['SVM'].fit(X_train, y_train)
            models['KNN'].fit(X_train, y_train)
        
        # Voting Classifier (Ensemble)
        if len(models) >= 2:  # Only create ensemble if we have at least 2 models
            voting_estimators = []
            if 'Logistic Regression' in models:
                voting_estimators.append(('lr', models['Logistic Regression']))
            if 'Random Forest' in models:
                voting_estimators.append(('rf', models['Random Forest']))
            if 'SVM' in models:
                voting_estimators.append(('svm', models['SVM']))
            if 'KNN' in models:
                voting_estimators.append(('knn', models['KNN']))
            
            if len(voting_estimators) >= 2:
                voting_clf = VotingClassifier(
                    estimators=voting_estimators,
                    voting='soft'
                )
                voting_clf.fit(X_train, y_train)
                models['Voting Classifier'] = voting_clf
        
        # Stacking Classifier (Advanced Ensemble)
        if len(models) >= 3:  # Only create stacking if we have at least 3 models
            stacking_estimators = []
            if 'Logistic Regression' in models:
                stacking_estimators.append(('lr', models['Logistic Regression']))
            if 'Random Forest' in models:
                stacking_estimators.append(('rf', models['Random Forest']))
            if 'SVM' in models:
                stacking_estimators.append(('svm', models['SVM']))
            
            if len(stacking_estimators) >= 2:
                stacking_clf = StackingClassifier(
                    estimators=stacking_estimators,
                    final_estimator=LogisticRegression(),
                    cv=3  # Reduced CV folds for better performance
                )
                stacking_clf.fit(X_train, y_train)
                models['Stacking Classifier'] = stacking_clf
        
        return models

    # Evaluate models
    def evaluate_models(models, X_test, y_test, X_train, y_train):
        results = {}
        
        # Initialize cross-validation with reduced folds for better performance
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, model in models.items():
            # Standard evaluation on test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std(),
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            }
        
        return results

    # Create tabs for different sections with icons
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Dataset Overview", 
        "⚙️ Feature Engineering", 
        "🤖 Model Training", 
        "📊 Model Evaluation", 
        "🔮 Prediction"
    ])

    # Tab 1: Dataset Overview
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("📋 Dataset Overview")
        
        # Load data
        df = load_data()
        df_engineered = engineer_features(df)
        df_processed = prepare_data(df_engineered)
        
        st.markdown("""
        <div style="background: linear-gradient(90deg, #3498db, #2c3e50); color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3>📊 Dataset Information</h3>
            <p>The PIMA Indian Diabetes Database is a widely used dataset for diabetes prediction research. 
            It contains medical data from female patients of Pima Indian heritage, a population with a high prevalence of diabetes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Size", f"{df.shape[0]} rows")
        with col2:
            st.metric("Features", f"{df.shape[1]-1} input")
        with col3:
            st.metric("Target Variable", "Diabetes Outcome")
        
        st.markdown("""
        <div class="card">
            <h4>🔤 Feature Descriptions</h4>
            <ul>
                <li><strong>Pregnancies</strong>: Number of times pregnant</li>
                <li><strong>Glucose</strong>: Plasma glucose concentration</li>
                <li><strong>BloodPressure</strong>: Diastolic blood pressure (mm Hg)</li>
                <li><strong>SkinThickness</strong>: Triceps skin fold thickness (mm)</li>
                <li><strong>Insulin</strong>: 2-Hour serum insulin (mu U/ml)</li>
                <li><strong>BMI</strong>: Body mass index (weight in kg/(height in m)^2)</li>
                <li><strong>DiabetesPedigreeFunction</strong>: Diabetes pedigree function</li>
                <li><strong>Age</strong>: Age in years</li>
                <li><strong>Outcome</strong>: Class variable (0 or 1)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 First 10 rows of the dataset")
        st.dataframe(df.head(10))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Class Distribution")
        class_dist = df['Outcome'].value_counts()
        st.bar_chart(class_dist)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Feature Statistics")
        st.dataframe(df.describe())
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Correlation matrix
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔗 Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=16)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Distribution plots
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📉 Feature Distributions")
        numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Create subplots for distributions
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, feature in enumerate(numeric_features):
            axes[i].hist(df[feature], bins=30, alpha=0.7, color='#3498db')
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Box plots for outlier detection
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📦 Box Plots for Outlier Detection")
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, feature in enumerate(numeric_features):
            box_plot = axes[i].boxplot(df[feature], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('#3498db')
            axes[i].set_title(f'{feature} Box Plot')
            axes[i].set_ylabel(feature)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: Feature Engineering
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("⚙️ Feature Engineering")
        
        df = load_data()
        df_engineered = engineer_features(df)
        
        st.markdown("""
        <div style="background: linear-gradient(90deg, #27ae60, #219653); color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3>🧬 Advanced Feature Engineering</h3>
            <p>We have created several derived features to improve model performance through domain knowledge and mathematical transformations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
            <h4>📊 Categorical Features</h4>
            <ul>
                <li><strong>BMI Category</strong>: Underweight, Normal, Overweight, Obese</li>
                <li><strong>Age Group</strong>: Young, Middle, Senior</li>
                <li><strong>Glucose Category</strong>: Normal, Prediabetic, Diabetic</li>
                <li><strong>Pregnancies Bin</strong>: None, Low, Medium, High</li>
                <li><strong>Skin Thickness Bin</strong>: Low, Normal, High, VeryHigh</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h4>🔢 Polynomial Features</h4>
            <ul>
                <li><strong>Glucose_Squared</strong>: Square of Glucose level</li>
                <li><strong>BMI_Squared</strong>: Square of BMI</li>
                <li><strong>Age_Squared</strong>: Square of Age</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h4>📊 Log Transformations</h4>
            <ul>
                <li><strong>Log_Insulin</strong>: Log of Insulin</li>
                <li><strong>Log_DiabetesPedigreeFunction</strong>: Log of Diabetes Pedigree Function</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>🔗 Interaction Features</h4>
            <ul>
                <li><strong>BMI_Age</strong>: Product of BMI and Age</li>
                <li><strong>Glucose_BMI</strong>: Product of Glucose and BMI</li>
                <li><strong>Insulin_Glucose</strong>: Product of Insulin and Glucose</li>
                <li><strong>Glucose_BloodPressure_Ratio</strong>: Ratio of Glucose to Blood Pressure</li>
                <li><strong>BMI_Age_Ratio</strong>: Ratio of BMI to Age</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h4>➗ Ratio Features</h4>
            <ul>
                <li><strong>BMI_Ratio</strong>: BMI divided by Age</li>
                <li><strong>Glucose_Ratio</strong>: Glucose divided by BMI</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
            <h4>🧮 Composite Features</h4>
            <ul>
                <li><strong>Health_Score</strong>: Composite health indicator</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📊 Distribution of Derived Features")
        
        # Show distribution of categorical features
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        categorical_features = ['BMI_Category', 'Age_Group', 'Glucose_Category', 'Pregnancies_Bin', 'SkinThickness_Bin']
        
        for i, feature in enumerate(categorical_features):
            if feature in df_engineered.columns:
                df_engineered[feature].value_counts().plot(kind='bar', ax=axes[i], color='#27ae60')
                axes[i].set_title(f'{feature} Distribution')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(categorical_features), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔗 Interaction Features Visualization")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        interaction_features = [
            ('BMI', 'Age'),
            ('Glucose', 'BMI'),
            ('Insulin', 'Glucose'),
            ('Glucose', 'BloodPressure'),
            ('BMI', 'Age')
        ]
        
        for i, (x_feature, y_feature) in enumerate(interaction_features):
            if i < len(axes):
                axes[i].scatter(df_engineered[x_feature], df_engineered[y_feature], c=df_engineered['Outcome'], alpha=0.5)
                axes[i].set_xlabel(x_feature)
                axes[i].set_ylabel(y_feature)
                axes[i].set_title(f'{x_feature} vs {y_feature} (colored by Outcome)')
        
        # Hide unused subplots
        for j in range(len(interaction_features), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

    # Tab 3: Model Training
    with tab3:
        st.header("Model Training")
        
        # Load and process data
        df = load_data()
        df_engineered = engineer_features(df)
        df_processed = prepare_data(df_engineered)
        
        # Prepare features and target
        # Select numerical features and encode categorical ones
        feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                          'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Age', 'Glucose_BMI', 
                          'Insulin_Glucose', 'Glucose_Squared', 'BMI_Squared', 'Age_Squared',
                          'BMI_Ratio', 'Glucose_Ratio', 'Log_Insulin', 'Log_DiabetesPedigreeFunction',
                          'Glucose_BloodPressure_Ratio', 'BMI_Age_Ratio', 'Health_Score']
        
        # Add encoded categorical features using one-hot encoding
        df_model = pd.get_dummies(df_processed, columns=['BMI_Category', 'Age_Group', 'Glucose_Category', 
                                                        'Pregnancies_Bin', 'SkinThickness_Bin'])
        
        # Update feature columns with encoded features
        feature_columns = [col for col in df_model.columns if col != 'Outcome']
        
        X = df_model[feature_columns]
        y = df_model['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=25)  # Increased k to accommodate more features
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Store in session state for use in other tabs
        st.session_state.X_train = X_train_selected
        st.session_state.X_test = X_test_selected
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.selector = selector
        st.session_state.feature_columns = feature_columns
        st.session_state.feature_names = X.columns.tolist()  # Store original feature names
        
        # Hyperparameter tuning option
        st.subheader("Training Options")
        hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning (takes longer)", value=False)
        
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a moment."):
                models = train_models(X_train_selected, y_train, hyperparameter_tuning)
                st.session_state.models = models
                
                # Feature importance for Random Forest
                if 'Random Forest' in models:
                    # Get feature names after selection
                    selected_features = selector.get_support(indices=True)
                    original_features = np.array(X.columns)[selected_features]
                    
                    feature_importance = pd.DataFrame({
                        'feature': original_features,
                        'importance': models['Random Forest'].feature_importances_
                    }).sort_values('importance', ascending=False)
                    st.session_state.feature_importance = feature_importance.head(15)
                
                st.success("Models trained successfully!")
                
                # Display results
                st.subheader("Training Results")
                for name, model in models.items():
                    st.write(f"{name} trained successfully")

    # Tab 4: Model Evaluation
    with tab4:
        st.header("Model Evaluation")
        
        if 'models' not in st.session_state:
            st.warning("Please train models first in the 'Model Training' tab.")
        else:
            # Evaluate models
            results = evaluate_models(st.session_state.models, st.session_state.X_test, st.session_state.y_test, 
                                    st.session_state.X_train, st.session_state.y_train)
            
            # Display metrics
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].round(4))
            
            # Visualization of metrics
            st.subheader("Model Performance Comparison")
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            fig, ax = plt.subplots(figsize=(12, 6))
            metrics_df[metrics].plot(kind='bar', ax=ax)
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # ROC Curves
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(10, 8))
            for name, result in results.items():
                if result['Probabilities'] is not None:
                    fpr, tpr, _ = roc_curve(st.session_state.y_test, result['Probabilities'])
                    ax.plot(fpr, tpr, label=f"{name} (AUC = {result['ROC-AUC']:.3f})")
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend()
            st.pyplot(fig)
            
            # Confusion Matrices
            st.subheader("Confusion Matrices")
            
            # Dynamically create subplot grid based on number of models
            num_models = len(results)
            cols = 3
            rows = (num_models + cols - 1) // cols  # Ceiling division
            
            if num_models > 0:
                fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
                
                # Handle case where there's only one model (axes is not an array)
                if num_models == 1:
                    axes = [axes]
                elif rows == 1 or cols == 1:
                    axes = axes.ravel()
                else:
                    axes = axes.ravel()
                
                # Plot confusion matrices
                plotted_models = 0
                for i, (name, result) in enumerate(results.items()):
                    cm = confusion_matrix(st.session_state.y_test, result['Predictions'])
                    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
                    axes[i].set_title(f'{name}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                    plotted_models = i + 1
                
                # Hide unused subplots
                for j in range(plotted_models, len(axes)):
                    axes[j].set_visible(False)
                
                st.pyplot(fig)
            else:
                st.warning("No models available for confusion matrix visualization.")
            
            # Feature Importance (for Random Forest)
            if 'feature_importance' in st.session_state:
                st.subheader("Feature Importance (Random Forest)")
                st.bar_chart(st.session_state.feature_importance.set_index('feature'))

    # Tab 5: Prediction
    with tab5:
        st.header("Diabetes Prediction")
        
        if 'models' not in st.session_state:
            st.warning("Please train models first in the 'Model Training' tab.")
        else:
            st.subheader("Enter Patient Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
                glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=100)
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            
            with col2:
                insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
                bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=25)
            
            with col3:
                st.subheader("Select Model")
                model_names = list(st.session_state.models.keys())
                selected_model = st.selectbox("Choose a model for prediction:", model_names)
                
                # Create derived features
                bmi_age = bmi * age
                glucose_bmi = glucose * bmi
                insulin_glucose = insulin * glucose
                glucose_squared = glucose ** 2
                bmi_squared = bmi ** 2
                age_squared = age ** 2
                bmi_ratio = bmi / (age + 1)
                glucose_ratio = glucose / (bmi + 1)
                log_insulin = np.log(insulin + 1)
                log_diabetes_pedigree = np.log(diabetes_pedigree + 1)
                glucose_bp_ratio = glucose / (blood_pressure + 1)
                bmi_age_ratio = bmi / (age + 1)
                health_score = (glucose / 200) * 0.3 + (bmi / 50) * 0.2 + (age / 100) * 0.2 + (pregnancies / 20) * 0.1 + (blood_pressure / 150) * 0.2
            
            # Prepare input data
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree],
                'Age': [age],
                'BMI_Age': [bmi_age],
                'Glucose_BMI': [glucose_bmi],
                'Insulin_Glucose': [insulin_glucose],
                'Glucose_Squared': [glucose_squared],
                'BMI_Squared': [bmi_squared],
                'Age_Squared': [age_squared],
                'BMI_Ratio': [bmi_ratio],
                'Glucose_Ratio': [glucose_ratio],
                'Log_Insulin': [log_insulin],
                'Log_DiabetesPedigreeFunction': [log_diabetes_pedigree],
                'Glucose_BloodPressure_Ratio': [glucose_bp_ratio],
                'BMI_Age_Ratio': [bmi_age_ratio],
                'Health_Score': [health_score]
            })
            
            # Add categorical features (simplified for prediction)
            # BMI Category
            if bmi < 18.5:
                input_data['BMI_Category_Underweight'] = 1
                input_data['BMI_Category_Normal'] = 0
                input_data['BMI_Category_Overweight'] = 0
                input_data['BMI_Category_Obese'] = 0
            elif 18.5 <= bmi <= 25:
                input_data['BMI_Category_Underweight'] = 0
                input_data['BMI_Category_Normal'] = 1
                input_data['BMI_Category_Overweight'] = 0
                input_data['BMI_Category_Obese'] = 0
            elif 25 < bmi <= 30:
                input_data['BMI_Category_Underweight'] = 0
                input_data['BMI_Category_Normal'] = 0
                input_data['BMI_Category_Overweight'] = 1
                input_data['BMI_Category_Obese'] = 0
            else:  # bmi > 30
                input_data['BMI_Category_Underweight'] = 0
                input_data['BMI_Category_Normal'] = 0
                input_data['BMI_Category_Overweight'] = 0
                input_data['BMI_Category_Obese'] = 1
                
            # Age Group
            if age <= 30:
                input_data['Age_Group_Young'] = 1
                input_data['Age_Group_Middle'] = 0
                input_data['Age_Group_Senior'] = 0
            elif 30 < age <= 50:
                input_data['Age_Group_Young'] = 0
                input_data['Age_Group_Middle'] = 1
                input_data['Age_Group_Senior'] = 0
            else:  # age > 50
                input_data['Age_Group_Young'] = 0
                input_data['Age_Group_Middle'] = 0
                input_data['Age_Group_Senior'] = 1
                
            # Glucose Category
            if glucose <= 100:
                input_data['Glucose_Category_Normal'] = 1
                input_data['Glucose_Category_Prediabetic'] = 0
                input_data['Glucose_Category_Diabetic'] = 0
            elif 100 < glucose <= 140:
                input_data['Glucose_Category_Normal'] = 0
                input_data['Glucose_Category_Prediabetic'] = 1
                input_data['Glucose_Category_Diabetic'] = 0
            else:  # glucose > 140
                input_data['Glucose_Category_Normal'] = 0
                input_data['Glucose_Category_Prediabetic'] = 0
                input_data['Glucose_Category_Diabetic'] = 1
                
            # Pregnancies Bin
            if pregnancies == 0:
                input_data['Pregnancies_Bin_None'] = 1
                input_data['Pregnancies_Bin_Low'] = 0
                input_data['Pregnancies_Bin_Medium'] = 0
                input_data['Pregnancies_Bin_High'] = 0
            elif 0 < pregnancies <= 2:
                input_data['Pregnancies_Bin_None'] = 0
                input_data['Pregnancies_Bin_Low'] = 1
                input_data['Pregnancies_Bin_Medium'] = 0
                input_data['Pregnancies_Bin_High'] = 0
            elif 2 < pregnancies <= 5:
                input_data['Pregnancies_Bin_None'] = 0
                input_data['Pregnancies_Bin_Low'] = 0
                input_data['Pregnancies_Bin_Medium'] = 1
                input_data['Pregnancies_Bin_High'] = 0
            else:  # pregnancies > 5
                input_data['Pregnancies_Bin_None'] = 0
                input_data['Pregnancies_Bin_Low'] = 0
                input_data['Pregnancies_Bin_Medium'] = 0
                input_data['Pregnancies_Bin_High'] = 1
                
            # Skin Thickness Bin
            if skin_thickness <= 20:
                input_data['SkinThickness_Bin_Low'] = 1
                input_data['SkinThickness_Bin_Normal'] = 0
                input_data['SkinThickness_Bin_High'] = 0
                input_data['SkinThickness_Bin_VeryHigh'] = 0
            elif 20 < skin_thickness <= 30:
                input_data['SkinThickness_Bin_Low'] = 0
                input_data['SkinThickness_Bin_Normal'] = 1
                input_data['SkinThickness_Bin_High'] = 0
                input_data['SkinThickness_Bin_VeryHigh'] = 0
            elif 30 < skin_thickness <= 50:
                input_data['SkinThickness_Bin_Low'] = 0
                input_data['SkinThickness_Bin_Normal'] = 0
                input_data['SkinThickness_Bin_High'] = 1
                input_data['SkinThickness_Bin_VeryHigh'] = 0
            else:  # skin_thickness > 50
                input_data['SkinThickness_Bin_Low'] = 0
                input_data['SkinThickness_Bin_Normal'] = 0
                input_data['SkinThickness_Bin_High'] = 0
                input_data['SkinThickness_Bin_VeryHigh'] = 1
            
            # Ensure all features are present (set missing ones to 0)
            for col in st.session_state.feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Reorder columns to match training data
            input_data = input_data[st.session_state.feature_names]
            
            # Scale and select features
            input_scaled = st.session_state.scaler.transform(input_data)
            input_selected = st.session_state.selector.transform(input_scaled)
            
            # Make prediction
            if st.button("Predict"):
                model = st.session_state.models[selected_model]
                prediction = model.predict(input_selected)[0]
                probability = model.predict_proba(input_selected)[0][1] if hasattr(model, 'predict_proba') else None
                
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error(f"⚠️ The model predicts that the patient **has diabetes**.")
                else:
                    st.success(f"✅ The model predicts that the patient **does not have diabetes**.")
                    
                if probability is not None:
                    st.info(f"Probability of having diabetes: {probability:.2%}")
                    
                    # Visualize probability
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.barh(['Diabetes Probability'], [probability], color='red' if probability > 0.5 else 'green')
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    st.pyplot(fig)
                    
                st.subheader("Model Information")
                st.write(f"Model used: {selected_model}")
                
                # Show model performance
                if 'models' in st.session_state:
                    # Use pre-computed results if available to avoid re-computation
                    if 'evaluation_results' in st.session_state:
                        results = st.session_state.evaluation_results
                    else:
                        results = evaluate_models(st.session_state.models, st.session_state.X_test, st.session_state.y_test,
                                                st.session_state.X_train, st.session_state.y_train)
                        st.session_state.evaluation_results = results  # Store for future use
                    model_result = results[selected_model]
                    st.write(f"Model Accuracy: {model_result['Accuracy']:.2%}")
                    st.write(f"Model Precision: {model_result['Precision']:.2%}")
                    st.write(f"Model Recall: {model_result['Recall']:.2%}")
                    if 'CV_Mean' in model_result:
                        st.write(f"Cross-Validation Score: {model_result['CV_Mean']:.2%} (±{model_result['CV_Std']:.2%})")

    # Footer
    st.markdown("---")
    st.markdown("Developed based on the IEEE paper 'Machine Learning Based Diabetes Prediction Using the PIMA Indian Dataset' with enhancements")