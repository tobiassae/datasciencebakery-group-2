# Sales Forecasting for a Bakery Branch - Group 2 

## Repository Link

https://github.com/tobiassae/datasciencebakery-group-2

## Description
### Task
This project focuses on sales forecasting for a bakery branch, utilizing historical sales data spanning from July 1, 2013, to July 30, 2018, to inform inventory and staffing decisions. We aim to predict future sales for six specific product categories: Bread, Rolls, Croissants, Confectionery, Cakes, and Seasonal Bread. Our methodology integrates statistical and machine learning techniques, beginning with a baseline linear regression model to identify fundamental trends, and progressing to a sophisticated neural network designed to discern more nuanced patterns and enhance forecast precision. The initiative encompasses data preparation, crafting bar charts with confidence intervals for visualization, and fine-tuning models to assess their performance on test data from August 1, 2018, to July 30, 2019, using the Mean Absolute Percentage Error (MAPE) metric for each product category.

### Our approach:
**Handling Missing Values:** 
The numerical columns ('Temperatur', 'Windgeschwindigkeit', 'Bewoelkung', 'Wettercode', 'IsWeekend') in the dataset had missing values. These were imputed using a KNN Imputer, which replaces missing values with the mean of the k nearest neighbors.

**Model Architecture:** 
The model used a Long Short-Term Memory (LSTM) layer with 128 units and a ReLU activation function as the input layer. This was followed by a Dropout layer with a rate of 0.3 to prevent overfitting. The final layer was a Dense layer with a single output unit, which would predict the sales value.

**Model Compilation:** 
The model was compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function. The Mean Absolute Percentage Error (MAPE) was also tracked as a metric.

**Training and Validation:** The model was trained for 150 epochs with a batch size of 32. Early stopping was used to monitor the validation loss and stop the training if the validation loss did not improve for 20 epochs. The best model checkpoint was saved during training.

**Visualization:** The loss functions for the training and validation datasets were plotted to visualize the model's performance during training.

### Task Type

Neural Network

### Results Summary

-   **Best Model:** Bakery_lstm_v6_0.ipynb
-   **Evaluation Metric:** MAPE
-   **Result by Category** (Identifier):
    -   **Bread** (1): 20.9%
    -   **Rolls** (2): 18.6%
    -   **Croissant** (3): 19.4%
    -   **Confectionery** (4): 25.4%
    -   **Cake** (5): 17.2%
    -   **Seasonal Bread** (6): 50.1%

## Documentation

1.  [**Data Import and Preparation**](0_DataPreparation/)
3.  [**Dataset Characteristics (Barcharts)**](1_DatasetCharacteristics/)
4.  [**Baseline Model**](2_BaselineModel/)
5.  [**Model Definition and Evaluation**](3_Model/)
6.  [**Presentation**](4_Presentation/Bakery_sales_predictions.pptx)

## Cover Image

![grafik](CoverImage/mape_finalmodel.png)