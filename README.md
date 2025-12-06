
## Project File Structure
```
ml-statistical-gaming-analysis/
│
├── data/                          # Raw and processed data
│   ├── raw/                       # Original Kaggle dataset
│   └── processed/                 # Cleaned/feature-engineered data
│
├── notebooks/                     # Jupyter notebooks model training and evaluation
│   ├── models/                    # Saves evaluation report and trained model 
│   └── model_training.ipynb
│
├── src/                           # Python scripts for data process and transformation
│   └──data_preprocessing.py
│
├── analytics/                       # Model performance, plots, visualiztion and dashboard
│
├── report/                        # Final Documentation / reports 
│
├── requirements.txt               # Python dependencies
├── README.md                      # Documentation for project File Structure
└── .gitignore                     # Ignore cache, venv, etc.
```

##  Documentation

- **[Data Analytics Guide](./reports/DATA_ANALYTICS_GUIDE.md)**  
  Learn about the statistical analysis methods and insights derived from the gaming data.

- **[Data Transformation Guide](./reports/DATA_TRANSFORMATION_GUIDE.md)**  
  Step-by-step guide on cleaning, transforming, and preparing the data for modeling.

- **[ML Models Reports](./reports/ML_MODELS_REPORTS.md)**  
  Evaluation and comparison of machine learning models applied on the processed data.

- **[KF Implementation Reports](./reports/KF_Implementation.md)**  
  Detailed evaluation and performance comparison of Kalman Filter, Extended Kalman Filter, and Unscented Kalman Filter on the processed dataset.