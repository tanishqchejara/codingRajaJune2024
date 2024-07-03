import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template_string
import io
import base64

# Step 1: Data Collection
def load_fraud_data():
    # For this example, we'll create a synthetic dataset
    np.random.seed(42)
    n_samples = 10000
    df = pd.DataFrame({
        'amount': np.random.exponential(scale=100, size=n_samples),
        'time': np.random.randint(0, 24, size=n_samples),
        'day_of_week': np.random.randint(0, 7, size=n_samples),
        'is_foreign': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'is_online': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    })
    df['is_fraud'] = ((df['amount'] > 500) & (df['is_foreign'] == 1) & (df['is_online'] == 1)).astype(int)
    df['is_fraud'] = df['is_fraud'] | (np.random.random(n_samples) < 0.01)  # Add some random fraud
    return df

# Step 2: Data Preprocessing
def preprocess_fraud_data(df):
    print("Fraud Data Shape:", df.shape)
    print("\nFraud Data Info:")
    df.info()
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFraud Distribution:")
    print(df['is_fraud'].value_counts(normalize=True))
    return df

# Step 3: Feature Engineering
def engineer_features(df):
    df['amount_log'] = np.log1p(df['amount'])
    df['hour'] = df['time']
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

# Step 4: Exploratory Data Analysis (EDA)
def plot_fraud_eda(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.histplot(data=df, x='amount_log', hue='is_fraud', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Transaction Amounts')
    
    sns.countplot(data=df, x='hour', hue='is_fraud', ax=axes[0, 1])
    axes[0, 1].set_title('Fraud by Hour of Day')
    
    sns.countplot(data=df, x='day_of_week', hue='is_fraud', ax=axes[1, 0])
    axes[1, 0].set_title('Fraud by Day of Week')
    
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Heatmap')
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Step 5, 6, 7: Model Selection, Training, and Evaluation
def train_fraud_model(df):
    features = ['amount_log', 'hour', 'day_of_week', 'is_foreign', 'is_online', 'is_weekend']
    X = df[features]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    cm_img = base64.b64encode(img.getvalue()).decode()
    
    return model, scaler, cm_img

# Flask App
app = Flask(__name__)

# Load and preprocess data
fraud_df = load_fraud_data()
fraud_df = preprocess_fraud_data(fraud_df)
fraud_df = engineer_features(fraud_df)
fraud_eda_img = plot_fraud_eda(fraud_df)
fraud_model, fraud_scaler, fraud_cm_img = train_fraud_model(fraud_df)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1, h2 { color: #333; }
        form { margin-bottom: 20px; }
        input[type="number"] { width: 100px; margin-right: 10px; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>Fraud Detection System</h1>
    
    <h2>Detect Fraud</h2>
    <form id="fraud-detection-form">
        <label for="amount">Amount:</label>
        <input type="number" id="amount" name="amount" min="0" step="0.01" required>
        <label for="time">Time (0-23):</label>
        <input type="number" id="time" name="time" min="0" max="23" required>
        <label for="day_of_week">Day of Week (0-6):</label>
        <input type="number" id="day_of_week" name="day_of_week" min="0" max="6" required>
        <label for="is_foreign">Is Foreign (0/1):</label>
        <input type="number" id="is_foreign" name="is_foreign" min="0" max="1" required>
        <label for="is_online">Is Online (0/1):</label>
        <input type="number" id="is_online" name="is_online" min="0" max="1" required>
        <button type="submit">Detect Fraud</button>
    </form>
    <div id="fraud-result"></div>
    
    <h2>Fraud Detection EDA</h2>
    <img src="data:image/png;base64,{{ fraud_eda_img }}" alt="Fraud Detection EDA">
    
    <h2>Fraud Detection Model Evaluation</h2>
    <img src="data:image/png;base64,{{ fraud_cm_img }}" alt="Confusion Matrix">

    <script>
        document.getElementById('fraud-detection-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            fetch('/detect_fraud', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => response.json())
            .then(data => {
                const fraudResultDiv = document.getElementById('fraud-result');
                fraudResultDiv.innerHTML = `<h3>Fraud Detection Result:</h3>
                   <p>Probability of Fraud: ${(data.fraud_probability * 100).toFixed(2)}%</p>
                    <p>Prediction: ${data.fraud_probability > 0.5 ? 'Fraudulent' : 'Not Fraudulent'}</p>`;
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, fraud_eda_img=fraud_eda_img, fraud_cm_img=fraud_cm_img)

@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    data = request.json
    features = pd.DataFrame({
        'amount_log': [np.log1p(float(data['amount']))],
        'hour': [int(data['time'])],
        'day_of_week': [int(data['day_of_week'])],
        'is_foreign': [int(data['is_foreign'])],
        'is_online': [int(data['is_online'])],
        'is_weekend': [1 if int(data['day_of_week']) >= 5 else 0]
    })
    
    features_scaled = fraud_scaler.transform(features)
    fraud_probability = fraud_model.predict_proba(features_scaled)[0][1]
    
    return jsonify({
        'fraud_probability': float(fraud_probability)
    })

if __name__ == '__main__':
    print("Starting the Fraud Detection System...")
    app.run(debug=True)