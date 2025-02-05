# Dataset Characteristics
[Code dataexploration](dataexploration.ipynb)

## Dataset Overview
- Descriptive statistics for numerical columns
- Violin plot


## Missing Values
- `isna()` function from pandas
```
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f"Column '{column}' has {nan_count} NaN values.")
```

## Feature Distributions
- Histogram
- Q-Q plot
- Relationship between two categorical variables
```python
crosstab = pd.crosstab(df['KielerWoche'], df['Umsatz'])
print(crosstab)
```
- Contingency table
- Chi Square

## Correlations