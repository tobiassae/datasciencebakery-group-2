# Baseline Model
Linear regression model

## Feature Selection
- Created 'temperatur_binned' feature, taking the season into account (e.g. 18C might be cold in summer, but warm in winter)
```
def bin_temperature(row):
        month = row['Month']
        temperature = row['Temperatur']
        
        if month in [12, 1, 2]:  # Winter
            if temperature <= 0:
                return 'Very Cold'
            elif temperature <= 5:
                return 'Cold'
            elif temperature <= 10:
                return 'Mild'
            else:
                return 'Warm'
        elif month in [3, 4, 5]:  # Spring
            if temperature <= 10:
                return 'Cool'
            elif temperature <= 15:
                return 'Mild'
            elif temperature <= 25:
                return 'Warm'
            else:
                return 'Hot'
        elif month in [6, 7, 8]:  # Summer
            if temperature <= 15:
                return 'Cool'
            elif temperature <= 20:
                return 'Mild'
            elif temperature <= 30:
                return 'Warm'
            else:
                return 'Hot'
        else:  # Fall
            if temperature <= 10:
                return 'Cool'
            elif temperature <= 15:
                return 'Mild'
            elif temperature <= 25:
                return 'Warm'
            else:
                return 'Hot'

    data['Temperatur_binned'] = data.apply(bin_temperature, axis=1)
```

## Implementation
[Code linear regression](model_2.ipynb)

## Evaluation
Model Performance:
Mean Squared Error: 5623.92
R-squared Score: 0.6676
Model Performance:
Mean Squared Error: 5623.92
R-squared Score: 0.6676
Mean Absolute Percentage Error (MAPE): 33.17%