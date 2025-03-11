import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load trained models
lm_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Manually input new X values (features)
# Ensure these values match the feature order used in training
new_data = pd.DataFrame([
    {'Speed': 3.8, 'IMUPeak': 2.9, 'Mass': 61, 'IMUsf': 3.8},
    {'Speed': 4.9, 'IMUPeak': 2.8, 'Mass': 65, 'IMUsf': 2.7},
])

# Load actual Y labels if available (to check accuracy)
actual_Y_labels = np.array([2.7, 3.8])  # Replace with actual values

# Predict using models
Y_pred_lm = lm_model.predict(new_data)
Y_pred_rf = rf_model.predict(new_data)

# Define function to evaluate model performance
def evaluate_model(Y_actual, Y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(Y_actual, Y_pred)),
        'R^2': r2_score(Y_actual, Y_pred),
        'MAE': mean_absolute_error(Y_actual, Y_pred)
    }

# Check accuracy only if actual Y labels are available
if len(actual_Y_labels) == len(Y_pred_lm):
    lm_results = evaluate_model(actual_Y_labels, Y_pred_lm)
    rf_results = evaluate_model(actual_Y_labels, Y_pred_rf)
    
    print("\nLinear Regression Model Performance:", lm_results)
    print("Random Forest Model Performance:", rf_results)

# Save predictions
predictions_df = pd.DataFrame({
    'Actual GRFPeak': actual_Y_labels,
    'Linear Regression Prediction': Y_pred_lm,
    'Random Forest Prediction': Y_pred_rf
})
predictions_df.to_csv('predictions_manual.csv', index=False)

print("\nPredictions saved to predictions_manual.csv")
