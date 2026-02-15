# Heart Disease Classification using Machine Learning

## 1. Problem Statement
The objective of this project is to build and evaluate multiple machine learning classification models to predict the presence of heart disease based on patient medical attributes. The project also includes deployment of the trained models using a Streamlit web application for interactive evaluation.

---

## 2. Dataset Description
The dataset used for this project is the Heart Disease Dataset obtained from Kaggle.  
It contains 1025 records and 14 attributes.  
The target variable indicates whether a patient has heart disease (1) or not (0).

The dataset includes the following features:
- age  
- sex  
- chest pain type (cp)  
- resting blood pressure (trestbps)  
- cholesterol (chol)  
- fasting blood sugar (fbs)  
- resting ECG (restecg)  
- maximum heart rate achieved (thalach)  
- exercise induced angina (exang)  
- ST depression (oldpeak)  
- slope  
- number of major vessels (ca)  
- thalassemia (thal)  

---

## 3. Models Used and Evaluation Metrics

The following machine learning models were implemented:
- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

The models were evaluated using the following metrics:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.809756 | 0.929810 | 0.761905 | 0.914286 | 0.831169 | 0.630908 |
| Decision Tree | 0.985366 | 0.985714 | 1.000000 | 0.971429 | 0.985507 | 0.971151 |
| KNN | 0.863415 | 0.962905 | 0.873786 | 0.857143 | 0.865385 | 0.726935 |
| Naive Bayes | 0.829268 | 0.904286 | 0.807018 | 0.876190 | 0.840183 | 0.660163 |
| Random Forest (Ensemble) | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost (Ensemble) | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

---

## 4. Observations

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | Achieves good performance with high recall, indicating it correctly identifies most patients with heart disease. However, its overall accuracy is lower compared to tree-based models. |
| Decision Tree | Provides very high accuracy and precision but may be prone to overfitting due to its ability to memorize patterns in the training data. |
| KNN | Performs well when features are scaled properly and shows balanced precision and recall, though it is sensitive to the choice of K and data distribution. |
| Naive Bayes | Performs reasonably well despite its strong assumption of feature independence, making it a simple and efficient baseline model. |
| Random Forest (Ensemble) | Achieves perfect scores across all evaluation metrics, indicating very strong predictive capability due to ensemble learning, though it may suggest possible overfitting. |
| XGBoost (Ensemble) | Also achieves perfect performance, demonstrating the power of boosted decision trees and its effectiveness in handling structured tabular data. |

---

## 5. Streamlit Application
A Streamlit web application was developed to:
- Upload test dataset (CSV)
- Select machine learning model from dropdown
- Display evaluation metrics
- Display confusion matrix

---

## 6. How to Run the Project Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py

## 7. Deployment
The Streamlit web application is deployed and accessible at:

https://heart-disease-ml-app-5kutcnbvgbdvl6jajwxxgk.streamlit.app/

