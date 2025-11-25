
### Project File Structure
```
ml-statistical-gaming-analysis/
│
├── data/                          # Raw and processed data
│   ├── raw/                       # Original Kaggle dataset
│   └── processed/                 # Cleaned/feature-engineered data
│
├── notebooks/                     # Jupyter notebooks (optional)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/                           # Python scripts for modular code
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── utils.py
│
├── results/                       # Model performance, plots, logs
│   ├── metrics.csv
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── report/                        # Final report / markdown summary
│   └── final_report.md
│
├── requirements.txt               # Python dependencies
├── main.py                        # Entry point for running the pipeline
├── README.md                      # Documentation
└── .gitignore                     # Ignore cache, venv, etc.
```
