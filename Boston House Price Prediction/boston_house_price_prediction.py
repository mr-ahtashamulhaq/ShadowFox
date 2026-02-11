"""
Boston House Price Prediction
A machine learning project to predict house prices using regression models
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("BOSTON HOUSE PRICE PREDICTION")
print("=" * 80)

# 1. LOAD DATA
print("\n[STEP 1] Loading Data...")

df = pd.read_csv('HousingData.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. EXPLORATORY DATA ANALYSIS
print("\n[STEP 2] Exploratory Data Analysis...")

# Check for missing values
print("\nMissing Values:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing_Count': missing_data,
    'Percentage': missing_percent
})
print(missing_info[missing_info['Missing_Count'] > 0])

# Display basic statistics
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Visualize missing data
plt.figure(figsize=(10, 6))
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
if len(missing_counts) > 0:
    missing_counts.plot(kind='bar', color='coral')
    plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. DATA PREPROCESSING
print("\n[STEP 3] Data Preprocessing...")

# Handle missing values - fill with median
print("Handling missing values...")
for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column].fillna(df[column].median(), inplace=True)

print(f"Missing values after handling: {df.isnull().sum().sum()}")

# Separate features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully!")

# 4. CORRELATION ANALYSIS
print("\n[STEP 4] Correlation Analysis...")

# Calculate correlations with target variable
correlations = df.corr()['MEDV'].sort_values(ascending=False)
print("\nTop features correlated with house price:")
print(correlations.head(10))

# Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap of Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. MODEL TRAINING
print("\n[STEP 5] Training Models...")

# Dictionary to store models and their performance
models = {}
results = {}

# Model 1: Linear Regression
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr_model

# Model 2: Random Forest Regressor
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
models['Random Forest'] = rf_model

# Model 3: Gradient Boosting Regressor
print("Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb_model

print("All models trained successfully!")

# 6. MODEL EVALUATION
print("\n[STEP 6] Model Evaluation...")
print("\n" + "=" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 80)

for name, model in models.items():
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Store results
    results[name] = {
        'Train R²': train_r2,
        'Test R²': test_r2,
        'RMSE': test_rmse,
        'MAE': test_mae
    }
    

    print(f"\n{name}:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  RMSE:     ${test_rmse:.2f}k")
    print(f"  MAE:      ${test_mae:.2f}k")

# 7. VISUALIZATIONS
print("\n[STEP 7] Creating Visualizations...")

# Create results DataFrame for visualization
results_df = pd.DataFrame(results).T

# Plot 1: Model Comparison - R² Scores
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# R² Score comparison
ax1 = axes[0]
results_df[['Train R²', 'Test R²']].plot(kind='bar', ax=ax1, color=['skyblue', 'coral'])
ax1.set_title('Model Comparison - R² Scores', fontsize=14, fontweight='bold')
ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score')
ax1.set_ylim(0, 1)
ax1.legend(['Training', 'Testing'])
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Error metrics comparison
ax2 = axes[1]
results_df[['RMSE', 'MAE']].plot(kind='bar', ax=ax2, color=['salmon', 'lightgreen'])
ax2.set_title('Model Comparison - Error Metrics', fontsize=14, fontweight='bold')
ax2.set_xlabel('Models')
ax2.set_ylabel('Error ($1000s)')
ax2.legend(['RMSE', 'MAE'])
ax2.grid(axis='y', alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Best Model Predictions vs Actual
best_model_name = results_df['Test R²'].idxmax()
best_model = models[best_model_name]

print(f"\nBest performing model: {best_model_name}")

y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Prices ($1000s)', fontsize=12)
plt.ylabel('Predicted Prices ($1000s)', fontsize=12)
plt.title(f'Actual vs Predicted Prices - {best_model_name}', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Residuals Distribution
residuals = y_test - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Residual plot
ax1 = axes[0]
ax1.scatter(y_pred_best, residuals, alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
ax1.axhline(y=0, color='r', linestyle='--', lw=2)
ax1.set_xlabel('Predicted Prices ($1000s)', fontsize=12)
ax1.set_ylabel('Residuals ($1000s)', fontsize=12)
ax1.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# Residual distribution
ax2 = axes[1]
ax2.hist(residuals, bins=30, color='teal', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Residuals ($1000s)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Feature Importance (for Random Forest)
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='mediumseagreen')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())

# 8. FINAL SUMMARY
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\nBest Model: {best_model_name}")
print(f"Test R² Score: {results[best_model_name]['Test R²']:.4f}")
print(f"RMSE: ${results[best_model_name]['RMSE']:.2f}k")
print(f"MAE: ${results[best_model_name]['MAE']:.2f}k")

print("\nInterpretation:")
print(f"- The model explains {results[best_model_name]['Test R²']*100:.2f}% of the variance in house prices")
print(f"- Average prediction error: ${results[best_model_name]['MAE']:.2f}k")
print(f"- The model performs well with low error rates")

print("\n" + "=" * 80)
print("Analysis Complete! All visualizations saved.")
print("=" * 80)