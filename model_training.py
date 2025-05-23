import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# 1. Load Dataset
df = pd.read_csv('simulated_ev_battery_data.csv')

# Example: Inspect the dataset
print(df.head())

# 2. Select Features and Target
features = ['Cycle', 'Temperature', 'Voltage (V)', 'Current (A)']  # sample features
target = 'State of Health (%)'  

X = df[features]
y = df[target]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 6. Evaluate Model
y_pred = rf_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.4f}')
print(f'R² Score: {r2:.4f}')

# 7. Plot Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.xlabel('Sample')
plt.ylabel('Capacity / SoH')
plt.title('Random Forest Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Save Model and Scaler
joblib.dump(rf_model, 'battery_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
