# Data Transformation Summary

One-line summary: the dataset was prepared for modeling by preserving IDs, standardizing numeric features, encoding categorical variables as small integers, detecting (but not removing) outliers, and dropping exact duplicates.

### Feature-level changes (short table)

| Feature | Raw data (example) | Processed data | Why (short) |
|---|---:|---|---|
| PlayerID | integers (e.g. 9000) | unchanged (integer) | Keep as identifier/traceability (not a feature) |
| Age | 15–49 | scaled (z-scores) | Normalize scale for modeling |
| Gender | Male/Female/Other | label encoded (0–2) | Convert text to numeric for models |
| Location | Other/USA/Europe/Asia | label encoded (0–3) | Convert text to numeric for models |
| GameGenre | Strategy/Action/RPG/... | label encoded | Convert text to numeric for models |
| PlayTimeHours | 0.0–24.0 | scaled (z-scores) | Normalize across users |
| InGamePurchases | 0–100+ (skewed) | scaled (z-scores); outliers flagged | Reduce scale differences; flagged for your decision |
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


### Quick commands

Re-run preprocessing locally:
```bash
python src/data_preprocessing.py
```
