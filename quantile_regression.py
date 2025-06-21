import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# Load your slippage dataset
df = pd.read_csv("slippage_data.csv")

# Features and target variable
X = df[["spread", "order_size", "depth"]]  # Features
y = df["slippage"]                         # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train Quantile Regression model (quantile = 0.5 means median regression)
model = QuantileRegressor(quantile=0.5, alpha=0)
model.fit(X_train, y_train)
joblib.dump(scaler, 'slippage_scaler.pkl')
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics and model parameters
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
joblib.dump(model, 'quantile_regression_model.pkl')
# Predict slippage for a new order (example)
new_order = pd.DataFrame([[0.1, 100, 1500]], columns=["spread", "order_size", "depth"])
predicted_slippage = model.predict(new_order)
print(f"Predicted slippage for new data: {predicted_slippage}")
