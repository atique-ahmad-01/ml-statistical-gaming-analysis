# ML Models Report — EngagementLevel Prediction

## 1. Executive summary
This report documents training, evaluation and saved outputs for models trained to predict `EngagementLevel`. Models trained: Logistic Regression, Decision Tree, Random Forest, XGBoost. Feature selection: RandomForest feature importances (n_estimators=300), top 14 features used.

## 2. Data & preprocessing
- Source: data/processed/gaming_data_processed.csv
- Features: all columns except `PlayerID` and `EngagementLevel`
- Target: `EngagementLevel` (multi-class)
- Split: 75% train / 25% test, stratified, random_state=42

## 3. Feature selection
- Method: RandomForestClassifier(n_estimators=300, random_state=42)
- Top features 
- Feature importance image: analytics/models output/
![Top 10 Feature Importance (Random Forest)](../analytics/models%20output/Top%2010%20Feature%20Importance%20(Random%20Forest).png)

## 4. Models & hyperparameters
- Logistic Regression — multinomial, solver=lbfgs, max_iter=2000
- Decision Tree — max_depth=5, random_state=42
- Random Forest — n_estimators=300, max_depth=7, random_state=42
- XGBoost — objective=multi:softmax, num_class=len(y.unique()), n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9

## 5. Evaluation procedure
- Metrics: accuracy, per-class precision/recall/f1 (classification report), confusion matrix (heatmap)
- Visuals and reports saved under analytics/models output/

## 6. Results (populated from analytics/models output)
# Confusion matrices:

![Logistic CM](../analytics/models%20output/Logistic%20Regression%20CM.png)
![Decision Tree CM](../analytics/models%20output/Decision%20Tree%20CM.png)

![Random Forest CM](../analytics/models%20output/Random%20Forest%20CM.png)
![XGBoost CM](../analytics/models%20output/XGBoost%20CM.png)

# Model accuracy comparison:

![Accuracy Comparison](../analytics/models%20output/Model%20Accuracy%20Comparison.png)

