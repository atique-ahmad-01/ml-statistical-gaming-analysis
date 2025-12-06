"""
Data Preprocessing Module for Online Gaming Behavior Dataset

This module handles loading, cleaning, and preprocessing the online gaming dataset.
It includes functions for handling missing values, encoding categorical variables,
scaling numerical features, and detecting outliers.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing for gaming behavior dataset."""
    
    def __init__(self, raw_data_path: str):
        """
        Initialize the preprocessor with path to raw data.
        
        Args:
            raw_data_path: Path to the raw CSV file
        """
        self.raw_data_path = Path(raw_data_path)
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        # Optional dicts to hold ordinal mappings (e.g., {'EngagementLevel': {'Low':0,'Medium':1,'High':2}})
        self.ordinal_mappings: Dict[str, dict] = {}
        self.numerical_features = []
        self.categorical_features = []
        # Columns that are identifiers and should not be used as features or scaled
        self.id_columns = ['PlayerID']
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.df = pd.read_csv(self.raw_data_path)
            logger.info(f"Successfully loaded data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self) -> None:
        """Display basic information about the dataset."""
        if self.df is None:
            logger.warning("No data loaded. Call load_data() first.")
            return
        
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"\nShape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        print("\nFirst few rows:")
        print(self.df.head())
    
    def identify_feature_types(self) -> Tuple[list, list]:
        """
        Identify numerical and categorical features.
        
        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        if self.df is None:
            logger.warning("No data loaded. Call load_data() first.")
            return [], []
        
        # Detect numerical and categorical features
        self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude identifier columns from numerical features (e.g., PlayerID)
        self.numerical_features = [c for c in self.numerical_features if c not in self.id_columns]
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()

        logger.info(f"Numerical features: {self.numerical_features}")
        logger.info(f"Categorical features: {self.categorical_features}")

        return self.numerical_features, self.categorical_features
    
    def handle_missing_values(self, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: Strategy for handling missing values
                     ('mean', 'median', 'drop', 'forward_fill')
        
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        if self.df is None:
            logger.warning("No data loaded. Call load_data() first.")
            return None
        
        self.df_processed = self.df.copy()
        missing_count = self.df_processed.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("No missing values found in dataset")
            return self.df_processed
        
        logger.info(f"Found {missing_count} missing values. Using '{strategy}' strategy.")
        
        if strategy == 'mean':
            # Fill numerical columns with mean
            for col in self.numerical_features:
                if self.df_processed[col].isnull().any():
                    self.df_processed[col].fillna(self.df_processed[col].mean(), inplace=True)
            # Fill categorical columns with mode
            for col in self.categorical_features:
                if self.df_processed[col].isnull().any():
                    self.df_processed[col].fillna(self.df_processed[col].mode()[0], inplace=True)
        
        elif strategy == 'median':
            for col in self.numerical_features:
                if self.df_processed[col].isnull().any():
                    self.df_processed[col].fillna(self.df_processed[col].median(), inplace=True)
            for col in self.categorical_features:
                if self.df_processed[col].isnull().any():
                    self.df_processed[col].fillna(self.df_processed[col].mode()[0], inplace=True)
        
        elif strategy == 'drop':
            self.df_processed.dropna(inplace=True)
            logger.info(f"Rows after dropping NaN: {self.df_processed.shape[0]}")
        
        elif strategy == 'forward_fill':
            self.df_processed.fillna(method='ffill', inplace=True)
            self.df_processed.fillna(method='bfill', inplace=True)
        
        logger.info("Missing values handled successfully")
        return self.df_processed
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Returns:
            pd.DataFrame: Dataset with duplicates removed
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        
        initial_rows = self.df_processed.shape[0]
        self.df_processed.drop_duplicates(inplace=True)
        removed = initial_rows - self.df_processed.shape[0]
        
        logger.info(f"Removed {removed} duplicate rows")
        return self.df_processed
    
    # def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, list]:
    #     """
    #     Detect outliers in numerical features.
        
    #     Args:
    #         method: Method for outlier detection ('iqr' or 'zscore')
    #         threshold: Threshold for outlier detection
        
    #     Returns:
    #         Dict mapping feature names to list of outlier indices
    #     """
    #     if self.df_processed is None:
    #         self.df_processed = self.df.copy()
        
    #     outliers = {}
        
    #     if method == 'iqr':
    #         for col in self.numerical_features:
    #             Q1 = self.df_processed[col].quantile(0.25)
    #             Q3 = self.df_processed[col].quantile(0.75)
    #             IQR = Q3 - Q1
    #             lower_bound = Q1 - threshold * IQR
    #             upper_bound = Q3 + threshold * IQR
    #             outlier_indices = self.df_processed[(self.df_processed[col] < lower_bound) | 
    #                                                  (self.df_processed[col] > upper_bound)].index.tolist()
    #             if outlier_indices:
    #                 outliers[col] = outlier_indices
    #                 logger.info(f"Found {len(outlier_indices)} outliers in {col}")
        
    #     elif method == 'zscore':
    #         from scipy import stats
    #         for col in self.numerical_features:
    #             z_scores = np.abs(stats.zscore(self.df_processed[col]))
    #             outlier_indices = self.df_processed[z_scores > threshold].index.tolist()
    #             if outlier_indices:
    #                 outliers[col] = outlier_indices
    #                 logger.info(f"Found {len(outlier_indices)} outliers in {col}")
        
    #     return outliers

    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, list]:
        """
        Detect outliers in numerical features, skipping binary features like InGamePurchases.

        Args:
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Dict mapping feature names to list of outlier indices
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()

        outliers = {}

        # Features that MUST NOT be checked for outliers
        skip_features = ["InGamePurchases"]

        # Only process valid continuous numerical features
        features_to_check = [
            col for col in self.numerical_features
            if col not in skip_features
        ]

        if method == 'iqr':
            for col in features_to_check:
                Q1 = self.df_processed[col].quantile(0.25)
                Q3 = self.df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_indices = self.df_processed[
                    (self.df_processed[col] < lower_bound) |
                    (self.df_processed[col] > upper_bound)
                ].index.tolist()

                if outlier_indices:
                    outliers[col] = outlier_indices
                    logger.info(f"Found {len(outlier_indices)} outliers in {col}")

        elif method == 'zscore':
            from scipy import stats
            for col in features_to_check:
                z_scores = np.abs(stats.zscore(self.df_processed[col]))
                outlier_indices = self.df_processed[z_scores > threshold].index.tolist()

                if outlier_indices:
                    outliers[col] = outlier_indices
                    logger.info(f"Found {len(outlier_indices)} outliers in {col}")

        # Log skipped binary columns
        skipped = set(self.numerical_features) - set(features_to_check)
        if skipped:
            logger.info(f"Skipped outlier detection for: {', '.join(skipped)}")

        return outliers


    def create_derived_features(self) -> pd.DataFrame:
        """
        Create derived features used for modeling.

        Adds columns to self.df_processed:
          - purchases_per_session = InGamePurchases / max(1, SessionsPerWeek)
          - playtime_per_week = PlayTimeHours * SessionsPerWeek
          - achievement_rate = AchievementsUnlocked / max(1, PlayerLevel)

        Returns:
            pd.DataFrame: Dataset with new derived features
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()

        # Avoid division by zero by replacing 0 with 1 where appropriate
        sess = self.df_processed.get('SessionsPerWeek')
        lvl = self.df_processed.get('PlayerLevel')

        if 'InGamePurchases' in self.df_processed.columns and sess is not None:
            sessions_safe = sess.replace(0, 1)
            self.df_processed['purchases_per_session'] = (
                self.df_processed['InGamePurchases'] / sessions_safe
            )
        else:
            # default to zeros if columns missing
            self.df_processed['purchases_per_session'] = 0

        if 'PlayTimeHours' in self.df_processed.columns and sess is not None:
            self.df_processed['playtime_per_week'] = (
                self.df_processed['PlayTimeHours'] * self.df_processed['SessionsPerWeek']
            )
        else:
            self.df_processed['playtime_per_week'] = 0

        if 'AchievementsUnlocked' in self.df_processed.columns and lvl is not None:
            level_safe = lvl.replace(0, 1)
            self.df_processed['achievement_rate'] = (
                self.df_processed['AchievementsUnlocked'] / level_safe
            )
        else:
            self.df_processed['achievement_rate'] = 0

        logger.info("Derived features created: purchases_per_session, playtime_per_week, achievement_rate")
        return self.df_processed
    
    def encode_categorical_features(self, method: str = 'label', ordinal_mappings: dict = None) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            method: Encoding method ('label' for LabelEncoder, 'onehot' for OneHotEncoder)
            ordinal_mappings: optional dict of column -> mapping for explicit ordinal encoding
        
        Returns:
            pd.DataFrame: Dataset with encoded features
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()
        # Apply any explicit ordinal mappings first (so they won't be label-encoded)
        if ordinal_mappings:
            for col, mapping in ordinal_mappings.items():
                if col in self.df_processed.columns:
                    # Map textual categories to defined integers. Unmapped values remain unchanged.
                    self.df_processed[col] = self.df_processed[col].map(mapping).astype(pd.Int64Dtype())
                    # Remove from categorical_features so it isn't label-encoded below
                    if col in self.categorical_features:
                        self.categorical_features = [c for c in self.categorical_features if c != col]
                    # Save mapping for reference
                    self.ordinal_mappings[col] = mapping
                    logger.info(f"Applied ordinal mapping for '{col}': {mapping}")

        if method == 'label':
            for col in self.categorical_features:
                le = LabelEncoder()
                # fit on original column values to preserve mapping
                self.df_processed[col] = le.fit_transform(self.df_processed[col].astype(str))
                self.encoders[col] = le
                logger.info(f"Label encoded '{col}'")
        
        elif method == 'onehot':
            self.df_processed = pd.get_dummies(self.df_processed, columns=self.categorical_features, drop_first=True)
            logger.info(f"One-hot encoded {len(self.categorical_features)} categorical features")
        
        return self.df_processed
    
    # def scale_numerical_features(self) -> pd.DataFrame:
    #     """
    #     Scale numerical features using StandardScaler.
        
    #     Returns:
    #         pd.DataFrame: Dataset with scaled features
    #     """
    #     if self.df_processed is None:
    #         self.df_processed = self.df.copy()
        
    #     if self.numerical_features:
    #         # Ensure id columns are not scaled (they were removed from numerical_features)
    #         self.df_processed[self.numerical_features] = self.scaler.fit_transform(
    #             self.df_processed[self.numerical_features]
    #         )
    #         logger.info(f"Scaled {len(self.numerical_features)} numerical features")
        
    #     return self.df_processed
    
    def scale_numerical_features(self) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler, 
        but skip binary features like InGamePurchases.

        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        if self.df_processed is None:
            self.df_processed = self.df.copy()

        if self.numerical_features:

            # List of features that should NOT be scaled
            skip_features = ["InGamePurchases"]

            # Final list of numerical features to scale
            features_to_scale = [
                col for col in self.numerical_features 
                if col not in skip_features
            ]

            if features_to_scale:
                self.df_processed[features_to_scale] = self.scaler.fit_transform(
                    self.df_processed[features_to_scale]
                )
                logger.info(f"Scaled {len(features_to_scale)} numerical features (excluding binary features)")

            # Inform if any numerical features were skipped
            skipped = set(self.numerical_features) - set(features_to_scale)
            if skipped:
                logger.info(f"Skipped scaling for: {', '.join(skipped)}")

        return self.df_processed

    
    def preprocess_pipeline(self, strategy: str = 'mean', outlier_method: str = 'iqr',
                           categorical_method: str = 'label', scale: bool = True,
                           ordinal_mappings: dict = None,
                           create_features: bool = False) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.
        
        Args:
            strategy: Missing value handling strategy
            outlier_method: Outlier detection method
            categorical_method: Categorical encoding method
            scale: Whether to scale numerical features
        
        Returns:
            pd.DataFrame: Fully preprocessed dataset
        """
        logger.info("Starting preprocessing pipeline...")
        
        self.identify_feature_types()
        # Optionally create derived features before handling missing values and scaling
        if create_features:
            self.handle_missing_values(strategy=strategy)  # ensure columns exist
            self.create_derived_features()
            # Re-identify types since we added numeric features
            self.identify_feature_types()
        else:
            self.handle_missing_values(strategy=strategy)
        self.remove_duplicates()
        
        outliers = self.detect_outliers(method=outlier_method)
        logger.info(f"Detected outliers in {len(outliers)} features")
        # Pass any ordinal mappings (e.g., EngagementLevel) into encoding step
        self.encode_categorical_features(method=categorical_method, ordinal_mappings=ordinal_mappings)
        
        if scale:
            self.scale_numerical_features()
        
        logger.info("Preprocessing pipeline completed successfully")
        return self.df_processed
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed dataset to CSV file.
        
        Args:
            output_path: Path where to save the processed data
        """
        if self.df_processed is None:
            logger.warning("No processed data available. Run preprocessing pipeline first.")
            return
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.df_processed.to_csv(output_file, index=False)
            logger.info(f"Processed data saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    # Note: artifact persistence (saving scaler/encoders to disk) was removed by request.
    # The fitted scaler and encoders remain available in memory on the DataPreprocessor
    # instance (self.scaler and self.encoders) for downstream model training.
    
    def get_preprocessed_data(self) -> pd.DataFrame:
        """
        Get the preprocessed dataset.
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        return self.df_processed


def main():
    """Main function to demonstrate data preprocessing."""
    # Define paths (relative to script location)
    script_dir = Path(__file__).parent.parent
    raw_data_path = script_dir / "data/raw/online_gaming_behavior_dataset.csv"
    output_path = script_dir / "data/processed/gaming_data_processed.csv"
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(raw_data_path)
    
    # Load and explore data
    preprocessor.load_data()
    preprocessor.explore_data()
    
    # Run preprocessing pipeline
    # Apply ordinal mapping for EngagementLevel: Low->0, Medium->1, High->2
    ordinal_map = {'EngagementLevel': {'Low': 0, 'Medium': 1, 'High': 2}}
    processed_df = preprocessor.preprocess_pipeline(
        strategy='mean',
        outlier_method='iqr',
        categorical_method='label',
        scale=True,
        ordinal_mappings=ordinal_map,
        create_features=True
    )
    
    # Save processed data
    preprocessor.save_processed_data(output_path)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    main()
