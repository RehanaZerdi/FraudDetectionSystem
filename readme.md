# 💳 Fraud Detection Project

### Submitted by: **Rehana Begum**  
**Boston Institute of Technology**

---

## 📘 Project Overview
Financial fraud is a growing challenge in today’s digital world, causing significant financial losses to individuals and organizations.  
This project focuses on developing a **machine learning-based fraud detection system** that can identify fraudulent transactions with high accuracy while minimizing false alerts.

---

## 🎯 Objectives
- Detect fraudulent transactions using machine learning models  
- Reduce false positives and false negatives  
- Gain insights from transaction data through EDA and visualization  
- Estimate the financial impact of fraud detection  
- Build a **Streamlit web app** for real-time fraud prediction  

---

## 📊 Dataset Overview
- **Source:** Publicly available financial transaction dataset  
- **Features:** Transaction amount, time, type, and customer information  
- **Target:** Fraudulent (`1`) or Non-Fraudulent (`0`)  
- **Nature:** Imbalanced dataset (few fraud cases compared to non-fraud)

---

## ⚙️ Steps Involved

### 1. Data Loading & Initial Exploration
- Imported dataset using Pandas  
- Checked dataset shape, column types, and summary statistics  

### 2. Data Quality Assessment
- Removed duplicates and handled missing values  
- Verified and standardized data types  

### 3. Feature Engineering
- Created new features from timestamps  
- Encoded categorical variables and scaled numeric columns  

### 4. EDA & Visualization
- Explored distributions and relationships using Matplotlib, Seaborn, and Plotly  
- Identified trends between transaction type, amount, and fraud probability  

### 5. Model Building
- Trained models: **Logistic Regression**, **Random Forest**  
- Tuned hyperparameters using cross-validation  

### 6. Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- Random Forest performed best with strong recall and balanced predictions  

### 7. Estimating Business Impact
- Calculated financial gains from prevented frauds  
- Measured potential losses from false negatives/positives  

### 8. Streamlit App
- Developed a web interface for real-time fraud prediction  
- Allows users to input transaction details and receive predictions instantly  

---

## 🧠 Technologies Used
- **Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, plotly, scikit-learn  
- **Modeling:** Logistic Regression, Random Forest  
- **App Framework:** Streamlit  
- **IDE/Environment:** VS Code / Google Colab  

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/<RehanaZerdi>/fraud-detection-project.git
cd fraud-detection-project
