# Model Definition and Evaluation

## Model Selection
LSTM 
## Feature Engineering
One-hot encode categorical features
```
warengruppe_dummies = pd.get_dummies(data['Warengruppe'], prefix='Warengruppe')
day_dummies = pd.get_dummies(data['DayOfWeek'], prefix='Day')
month_dummies = pd.get_dummies(data['Month'], prefix='Month')
```

## Hyperparameter Tuning
- LSTM layer:
    Units: 64 (the number of units or neurons in the LSTM layer)
- Activation function: 
    'relu' (the activation function used in the LSTM layer)
- Input shape: 
    `(X_train_seq.shape[1], X_train_seq.shape[2])` (the shape of the input data)
- Dropout layer:
    Rate: 0.3 (the fraction of input units to drop)
- Dense layer:
    Units: 1 (the number of units or neurons in the output layer)

## Implementation
- Learning rate scheduler function

## Evaluation Metrics
MAPE by Warengruppe:
```
Warengruppe 2: 13.71%
Warengruppe 5: 15.34%
Warengruppe 3: 18.85%
Warengruppe 1: 20.19%
Warengruppe 4: 27.35%
Warengruppe 6: 51.54%
```