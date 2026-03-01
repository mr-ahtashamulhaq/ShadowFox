# ShadowFox AIML Internship Projects

This repository contains all the tasks completed as part of the ShadowFox AIML Internship. The work focuses on applying machine learning and data analysis techniques to real-world datasets, along with exploring modern language models.

The projects are organized by task levels: Beginner, Intermediate, and Advanced.

---

## Projects Overview

### 1. Boston House Price Prediction (Beginner)

A regression-based machine learning project to predict housing prices using multiple features such as crime rate, number of rooms, and other factors.

#### Key Work:
- Data preprocessing and cleaning
- Feature analysis and correlation study
- Training multiple regression models
- Model evaluation using RMSE, MAE, and R² score
- Performance comparison and visualization

#### Tech Used:
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

### 2. Store Sales and Profit Analysis (Intermediate)

A data analysis project focused on understanding sales and profit patterns in a retail dataset and extracting business insights.

#### Key Work:
- Sales and profit trend analysis
- Regional and category-level performance
- Customer segmentation
- Impact of discounts on profitability
- Business insights and recommendations

#### Tech Used:
- Python
- Pandas
- Matplotlib, Seaborn

---

### 3. LLM Analysis Notebook (Advanced)

An exploration and analysis of a pretrained language model using HuggingFace Transformers. This project focuses on understanding how language models behave under different prompts.

#### Key Work:
- Using `distilgpt2` for text generation
- Prompt-based experimentation (creative, factual, contextual)
- Analysis of model behavior (coherence, repetition, context)
- Token and lexical diversity analysis
- Visualization of generation statistics
- Research-based insights on model strengths and limitations

#### Tech Used:
- Python
- HuggingFace Transformers
- PyTorch
- Matplotlib, Seaborn

---

## Repository Structure

```

mr-ahtashamulhaq-shadowfox/
├── README.md
├── Boston House Price Prediction/
│   ├── README.md
│   ├── boston_house_price_prediction.py
│   ├── HousingData.csv
│   ├── requirements.txt
│   └── outputs/
│
├── Store Sales and Profit Analysis/
│   ├── README.md
│   ├── store_sales_profit_analysis.py
│   ├── requirements.txt
│   └── outputs/
│
└── LLM-Analysis-Notebook/
├── README.md
├── lm_analysis.ipynb
├── lm_analysis.py
└── outputs/

````

---

## Key Learnings

- Building end-to-end machine learning pipelines
- Working with real-world datasets and extracting insights
- Evaluating model performance using standard metrics
- Understanding how pretrained language models generate text
- Analyzing LLM outputs beyond just generation (behavior + limitations)

---

## How to Run Projects

Each project contains its own `requirements.txt`.

Example:

```bash
pip install -r requirements.txt
python filename.py
````

For the LLM project, you can also run the notebook in Google Colab.

---

## Focus Area

This repository reflects a focus on:

* Practical machine learning
* Data-driven analysis
* Applying pretrained models instead of training from scratch

---

## Future Improvements

* Adding model deployment (Flask / Streamlit)
* Comparing multiple ML models in each task
* Extending LLM project with API-based models (OpenAI, etc.)
* Improving visualizations and dashboards

---

## Author

Ahtasham Ul Haq
ShadowFox AIML Intern