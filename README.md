# Credit Card Default Prediction using XGBoost

This project leverages the **XGBoost Classifier** to predict whether a credit card holder will default on their payment. The model is trained using customer attributes, and a **Streamlit UI** is provided for easy interaction and predictions.

## Project Structure

```
credit_card_default/
│── data/
│   ├── default_of_credit_card_clients.csv
│── model/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│── train_model.py  # Train the XGBoost model
│── predict_ui.py   # Streamlit UI for making predictions
│── requirements.txt  # Dependencies
│── venv/  # Virtual environment (not included in repo)
│── .gitignore  # Ignore unnecessary files
│── README.md  # Project documentation
```

## Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/credit-card-default.git
cd credit-card-default
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model

```bash
python train_model.py
```

This will train an **XGBoost Classifier**, save the trained model, scaler, and feature names.

### 5️⃣ Run the Streamlit UI

```bash
streamlit run predict_ui.py
```

This will launch a **web-based interface** where users can enter input values and get predictions.

## How It Works

1. **Model Training:** `train_model.py` loads the dataset, preprocesses it, and trains an **XGBoost Classifier**.
2. **Feature Scaling:** A **StandardScaler** is used to normalize input features.
3. **Prediction UI:** `predict_ui.py` provides an interface where users can **paste comma-separated values** for prediction.
4. **Confidence Score:** The model outputs the probability of default along with its prediction.

## Example Input & Prediction

Paste the following in the UI:

```
50000,1,2,1,57,-1,0,-1,0,0,0,8617,5670,35835,20940,19146,19131,2000,36681,10000,9000,689,679,0
```

### Output Example:

```
⚠️ Likely to Default (Confidence: 79.5%)
```

or

```
✅ No Default Expected (Confidence: 87.3%)
```

## Technologies Used

- **Python**
- **XGBoost** (Extreme Gradient Boosting Classifier)
- **scikit-learn** (Data Preprocessing)
- **pandas** (Data Handling)
- **numpy** (Numerical Computations)
- **Streamlit** (UI for Predictions)
- **joblib** (Saving Model & Scaler)

## To-Do / Improvements

- 🔹 Experiment with hyperparameter tuning to improve accuracy
- 🔹 Compare performance with other models (e.g., **Logistic Regression**, **Random Forest**)
- 🔹 Deploy as an API using Flask or FastAPI

## Contributing

Feel free to **fork** this repository, make improvements, and submit a **pull request**! 🚀

