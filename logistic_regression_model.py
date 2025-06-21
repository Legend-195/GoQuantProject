import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load the CSV
df = pd.read_csv("maker_taker_data.csv")

# 2. Clean missing values
df.dropna(inplace=True)

# 3. Encode the label (maker = 0, taker = 1)
df['maker_taker'] = df['maker_taker'].map({'maker': 0, 'taker': 1})
df['order_type_encoded'] = df['order_type'].map({'limit': 0, 'market': 1})
# 🛠 Ensure both classes exist — if not, add dummy sample
if df['maker_taker'].nunique() == 1:
    only_class = df['maker_taker'].unique()[0]
    dummy_class = 1 - only_class  # Add the missing class

    # Dummy feature values (reasonable defaults)
    dummy_row = {
        "spread": 0.5,
        "imbalance": 0.0,
        "volatility": 12.5,
        "order_size_usd": 200,
        "order_type":'limit',
        "order_type_encoded":0,
        "maker_taker": dummy_class
    }

    df = pd.concat([df, pd.DataFrame([dummy_row])], ignore_index=True)
    print("⚠️ Added dummy sample for missing class to allow training.")

# 4. Features and labels
X = df[["spread", "imbalance", "volatility", "order_size_usd","order_type_encoded"]]
y = df["maker_taker"]

# 5. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 7. Train logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Evaluate performance
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# 9. Predict on new input
new_order = [[0.1, 0.1, 12.5, 1000,1]]  # spread, imbalance, volatility, USD
new_order_scaled = scaler.transform(new_order)
prob = model.predict_proba(new_order_scaled)[0]

# Save model and scaler
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'logistic_scaler.pkl')

# Print probabilities
print(f"🔮 Predicted probabilities → Maker: {prob[0]:.2f}, Taker: {prob[1]:.2f}")
