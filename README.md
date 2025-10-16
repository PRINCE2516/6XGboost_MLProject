# 6XGboost_MLProject

🧠 Wholesale Customers Segmentation using XGBoost
📌 Project Overview

This project focuses on classifying Wholesale Customers into Horeca (Hotel/Restaurant/Café) or Retail categories using the XGBoost algorithm.
The model analyzes customers’ purchasing behavior to understand which features (like Grocery, Milk, Fresh, etc.) influence customer segmentation.

🎯 Objective

To build an accurate and interpretable machine learning model using XGBoost that predicts the customer channel type based on purchasing data.

⚙️ Workflow Summary
1. Importing Libraries

Essential Python libraries were imported:

pandas, numpy → Data manipulation

matplotlib, seaborn → Data visualization (EDA)

sklearn → Data splitting and evaluation metrics

xgboost → Main ML algorithm

2. Loading the Dataset

Dataset: Wholesale customers data.csv
It includes customer purchase data for various product categories (Milk, Grocery, Detergents, etc.) and their Channel (target label).

3. Exploratory Data Analysis (EDA)

Performed to:

Understand feature distributions

Check missing values

Detect outliers

Visualize relationships between features

EDA ensures data quality and helps identify which features are likely to be important.

4. Feature and Target Split

Features (X): All columns except Channel

Target (y): Channel (customer type)

This separation allows the model to learn the mapping X → y.

5. Train–Test Split

Data is split into:

Training Set (80%) → Used to train the model

Testing Set (20%) → Used to evaluate model performance

This ensures that the model generalizes well on unseen data.

6. Model Training with XGBoost

XGBoost is an ensemble-based gradient boosting algorithm.

It trains multiple decision trees sequentially — each new tree corrects errors made by previous ones.

Regularization (L1 & L2) helps prevent overfitting.

Key parameters used:

max_depth = 4
alpha = 10
learning_rate = 1.0
colsample_bytree = 0.3

7. Baseline Model Evaluation

Model was evaluated using:

accuracy_score(y_test, y_pred)


Baseline Accuracy: ~91.67%

8. k-Fold Cross Validation

To validate model consistency, 3-fold cross-validation was applied using:

xgboost.cv()


This provided more reliable metrics like AUC, ensuring the model is stable across data splits.

9. Feature Importance

Feature importance plot was generated to identify which variables most influenced predictions.

Top influential features:

Grocery

Detergents_Paper

Milk

10. Final Evaluation

Final model performance was assessed using:

Accuracy Score

AUC Score

Cross-validation metrics

The model achieved high accuracy and strong generalization on unseen data.

📈 Results Summary
Metric	Value
Accuracy	91.67%
AUC (Cross-Validation)	High (Consistent Performance)
Important Features	Grocery, Milk, Detergents_Paper
🧩 Key Learnings

XGBoost provides speed, accuracy, and interpretability.

Regularization (L1 & L2) helps control overfitting.

Cross-validation ensures robust performance evaluation.

Understanding feature importance improves business insight.

project-root/
  data/
    Wholesale customers data.csv
  notebooks/
    XGBoost_Model.ipynb
  src/
    model.py
  results/
    feature_importance.png
  README.md
  requirements.txt



🧰 Technologies Used

Python 3.x

XGBoost

NumPy / Pandas

Matplotlib / Seaborn

Scikit-learn

🚀 Future Enhancements

Apply GridSearchCV for deeper hyperparameter tuning

Try other models (Random Forest, LightGBM) for comparison

Deploy model using Streamlit or Flask
