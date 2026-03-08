# 🛡 FraudShield Pro — Oracle Insurance Fraud Detection

## 📁 Project Structure
```
project/
├── app.py
├── requirements.txt
├── fraud_oracle.csv          ← place dataset here
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   ├── model_results.pkl
│   └── best_model_name.pkl
├── notebooks/
│   └── Fraud_Oracle_Complete.ipynb
└── templates/
    └── index.html
```

## 🚀 How to Run
```bash
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

## 📊 Model Performance
| Model               | Accuracy |
|---------------------|----------|
| Decision Tree ⭐    | 94.13%   |
| Random Forest       | 94.07%   |
| Logistic Regression | 94.00%   |
| KNN                 | 94.00%   |
| SVM                 | 94.00%   |
| Naive Bayes         | 84.95%   |

## Dataset
- **Source:** fraud_oracle.csv
- **Records:** 15,420
- **Features:** 33 (30 after preprocessing)
- **Fraud Rate:** 6.0% (923 cases)
