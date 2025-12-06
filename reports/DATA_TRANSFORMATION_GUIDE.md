# Data Transformation Summary

 The dataset was prepared for modeling by preserving IDs, standardizing numeric features, encoding categorical variables as small integers, detecting (but not removing) outliers, and dropping exact duplicates.

### Feature-level changes (short table)

| Feature | Raw data (example) | Processed data | Why (short) |
|---|---:|---|---|
| PlayerID | integers (e.g. 9000) | unchanged (integer) | Keep as identifier/traceability (not a feature) |
| Age | 15–49 | scaled (z-scores) | Normalize scale for modeling |
| Gender | Male/Female/Other | label encoded (0–2) | Convert text to numeric for models |
| Location | Other/USA/Europe/Asia | label encoded (0–3) | Convert text to numeric for models |
| GameGenre | Strategy/Action/RPG/... | label encoded | Convert text to numeric for models |
| PlayTimeHours | 0.0–24.0 | scaled (z-scores) | Normalize across users |
| InGamePurchases | 0/1 (binary) | no scaling; keep as-is | Binary feature — scaling not needed and outliers not applicable |
| GameDifficulty | Easy/Medium/Hard | label encoded | Convert text to numeric (no ordinal assumption) |
| SessionsPerWeek | 0–20+ | scaled (z-scores) | Normalize count features |
| AvgSessionDurationMinutes | 0–180+ | scaled (z-scores) | Normalize duration for modeling |
| PlayerLevel | 0–100+ | scaled (z-scores) | Normalize progression levels |
| AchievementsUnlocked | 0–100+ | scaled (z-scores) | Normalize counts for modeling |
| EngagementLevel | Low/Medium/High | ordinal mapped (Low→0, Medium→1, High→2) | Explicit ordinal mapping applied for modeling |

### Notes

- `PlayerID` is preserved exactly and not used for scaling or as a numeric feature.
- Outliers are detected (IQR default) and logged; they are not removed automatically.
- Exact duplicate rows are dropped.
- Fitted scaler and label encoders are kept in memory on the `DataPreprocessor` instance (no files written to disk by default).

### Label encoding mappings

The exact mappings produced by the preprocessing (LabelEncoder alphabetical mapping) are:

- Gender:
	- Female -> 0
	- Male -> 1

- Location:
	- Asia -> 0
	- Europe -> 1
	- Other -> 2
	- USA -> 3

- GameGenre:
	- Action -> 0
	- RPG -> 1
	- Simulation -> 2
	- Sports -> 3
	- Strategy -> 4

- GameDifficulty:
	- Easy -> 0
	- Hard -> 1
	- Medium -> 2

### Simple formulas (what we did and why)

This short section shows the exact math behind the common transforms used in preprocessing. Each formula is one line and a tiny example.

- Z-score (standardization)
	- Formula: z = (x - μ) / σ
	- Why: puts numeric features on the same scale (mean≈0, std≈1) so models compare features fairly.
	- Example: Age = 43 → z ≈ (43 - mean_age) / std_age (≈ 1.09 in the processed data)

- Log 1-plus (reduce skew, keep zeros)
	- Formula: y = ln(1 + x)
	- Why: makes heavily skewed counts (like purchases) less extreme while preserving zeros.
	- Example: x = 8 purchases → y = ln(9) ≈ 2.20 (then we may standardize y)

- Winsorize / cap extremes
	- Formula (concept): y = min(max(x, P_low), P_high) where P_low and P_high are percentile cutoffs
	- Why: limits extreme values to reduce influence of outliers while keeping rank/order.
	- Example: cap values below 1st percentile or above 99th percentile to those percentile values.

- Label encoding (categorical → integer)
	- Formula (concept): map each distinct category to a unique integer (alphabetical by default for LabelEncoder)
	- Why: many models need numeric inputs; label encoding is compact but not ordinal unless you intend it.
	- Example: Gender: Female -> 0, Male -> 1

- Ordinal mapping (explicit order)
	- Formula (concept): map categories to integers according to a defined order (not alphabetical)
	- Why: use this when categories have a natural order (e.g., Low < Medium < High).
	- Example: EngagementLevel: Low -> 0, Medium -> 1, High -> 2

These formulas are intentionally simple. If you want me to apply one (for example `log1p` or `winsorize`) to `InGamePurchases` now, I can add that option to the pipeline and re-run the preprocessing so you can inspect before/after samples.

### Quick commands

Re-run preprocessing locally:
```bash
python src/data_preprocessing.py
```
