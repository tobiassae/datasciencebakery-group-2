# Data Preparation
## Data Importing
The CSV files were loaded into pandas dataframes using the `pd.read_csv()` function.

## Merging Data from different sources
The pandas dataframes were merged usinst the `pd.merge()` function.
```python
df = pd.merge(df_umsatz, df_wetter, on='Datum', how='left')
df = pd.merge(df, df_kiwo, on='Datum', how='left')
```

## Handling Missing Values
- In the [first versions](../1_DatasetCharacteristics/dataexploration.ipynb): `isna()` function from pandas was used
```
for column in df.columns:
    nan_count = df[column].isna().sum()
    print(f"Column '{column}' has {nan_count} NaN values.")
```

- For the [baseline model](../2_BaselineModel/model_2.ipynb)
```python
# More robust missing value handling
    numeric_columns = ['Bewoelkung', 'Windgeschwindigkeit', 'Wettercode']
    categorical_columns = ['Warengruppe', 'Temperatur_binned']

    # Fill numeric columns with median
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].median())
    
    # Fill categorical columns with mode
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
```
    
- In the [final model](//3_Model/bakery_lstm_6_0.ipynb): KNN Imputer was used to replace missing values 
```python
from sklearn.impute import KNNImputer
numerical_cols = ['Temperatur', 'Windgeschwindigkeit', 'Bewoelkung', 'Wettercode', 'IsWeekend']
knn_imputer = KNNImputer(n_neighbors=5)
data[numerical_cols] = knn_imputer.fit_transform(data[numerical_cols])
```