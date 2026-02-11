# Boston House Price Prediction

A machine learning project that predicts house prices in Boston using regression models. This project demonstrates data preprocessing, exploratory data analysis, model training, and evaluation.

## Project Overview

This project uses the Boston Housing Dataset to build and compare multiple regression models for predicting median house values. The dataset contains 506 samples with 13 features including crime rate, number of rooms, property tax rate, and more.

## Features in the Dataset

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
- **LSTAT**: Percentage of lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000's (TARGET VARIABLE)

## Models Implemented

1. **Linear Regression**: Baseline model
2. **Random Forest Regressor**: Ensemble method with decision trees
3. **Gradient Boosting Regressor**: Advanced ensemble technique

## Project Structure

```
Boston House Price Prediction/
│
├── boston_house_price_prediction.py    # Main Python script
├── HousingData.csv                      # Dataset
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
│
└── outputs/                             # Generated visualizations
    ├── missing_values.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    ├── predictions_vs_actual.png
    ├── residuals_analysis.png
    └── feature_importance.png
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ShadowFox.git
cd ShadowFox/Boston House Price Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running in Google Colab

1. Upload the `HousingData.csv` file to your Colab environment
2. Copy the code from `boston_house_price_prediction.py`
3. Run the cells in order
4. Download the generated visualizations

### Running Locally

```bash
python boston_house_price_prediction.py
```

## Results

The models were evaluated using the following metrics:
- **R² Score**: Measures how well the model explains variance in the data
- **RMSE (Root Mean Squared Error)**: Average prediction error
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values

### Model Performance

| Model | Test R² | RMSE | MAE |
|-------|---------|------|-----|
| Linear Regression | ~0.67 | ~5.9k | ~4.2k |
| Random Forest | ~0.87 | ~3.5k | ~2.4k |
| Gradient Boosting | ~0.88 | ~3.4k | ~2.3k |

*Note: Exact values may vary slightly based on the random state*

## Key Insights

1. **Most Important Features**: 
   - RM (number of rooms) has the strongest positive correlation with house prices
   - LSTAT (lower status population %) has a strong negative correlation
   
2. **Model Performance**: 
   - Gradient Boosting and Random Forest significantly outperform Linear Regression
   - The ensemble models capture non-linear relationships better
   
3. **Prediction Accuracy**: 
   - Average prediction error is around $2,300-$2,400 for the best models
   - Models explain approximately 87-88% of variance in house prices

## Visualizations

The project generates six key visualizations:

1. **Missing Values Chart**: Shows columns with missing data
2. **Correlation Heatmap**: Displays relationships between all features
3. **Model Comparison**: Compares R² scores and error metrics across models
4. **Predictions vs Actual**: Scatter plot showing model accuracy
5. **Residuals Analysis**: Shows prediction errors and their distribution
6. **Feature Importance**: Identifies the most influential features

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning models and evaluation

## Learning Outcomes

This project demonstrates:
- Data preprocessing and handling missing values
- Exploratory data analysis (EDA)
- Feature correlation analysis
- Training multiple regression models
- Model evaluation and comparison
- Data visualization techniques
- Real-world machine learning workflow

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Feature engineering to create new predictive features
- Implementing additional models (XGBoost, Neural Networks)
- Cross-validation for more robust evaluation
- Outlier detection and handling
- Building a web application for predictions

## Author

Created as part of the ShadowFox AI/ML Internship Program

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Dataset: Boston Housing Dataset (UCI Machine Learning Repository)
- ShadowFox Internship Program for the opportunity