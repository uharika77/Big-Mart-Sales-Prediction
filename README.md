# Big Mart Sales Prediction

## Overview

This project aims to predict the sales of products in different Big Mart outlets based on historical sales data. We use various machine learning techniques to build a predictive model.

## Dataset

The dataset consists of two files: train.csv and test.csv.

- train.csv: Contains historical sales data including product and outlet information.
- test.csv: Contains similar information but without sales figures, which are to be predicted.
  
## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Data Preprocessing

1.Loading Data: Read the train and test datasets using pandas.

2.Handling Missing Values:

- Item_Weight: Filled with the mean weight.
- Outlet_Size: Filled with "Unknown".
  
3.Categorical Encoding:

- One-hot encoding was applied to categorical variables like Item_Fat_Content, Item_Type, Outlet_Size, Outlet_Location_Type, and Outlet_Type.
  
4.Feature Engineering:

- Created a new feature Years_of_Operation based on Outlet_Establishment_Year.
- Replaced zero values in Item_Visibility with the mean visibility.
  
## Exploratory Data Analysis

- Visualized distributions of Item_Type, Item_Fat_Content, Outlet_Size, and other features.
- Plotted relationships between features and the target variable Item_Outlet_Sales.
  
## Model Building

1.Linear Regression
- Training: Fit a linear regression model on the training data.
- Evaluation: Used metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to evaluate performance.

2.Random Forest
- Training: Fit a Random Forest Regressor with hyperparameter tuning using Grid Search.
- Evaluation: Evaluated using the same metrics as above.

3.XGBoost
- Training: Fit an XGBoost Regressor with hyperparameter tuning.
- Evaluation: Evaluated using the same metrics as above.

## Results

The performance of different models was compared using MAE, MSE, RMSE, and Explained Variance Score.

## Usage

- Clone the repository and navigate to the project directory.
- Ensure all dependencies are installed.
- Run the Jupyter notebook or Python script to train the model and make predictions on the test set.

git clone <repository-url>

cd <project-directory>

pip install -r requirements.txt

jupyter notebook Big_Mart_Sales_Prediction.ipynb

## Conclusion

This project demonstrates the application of machine learning techniques to predict sales, providing insights into feature importance and model performance.

## References

[Pandas Documentation](https://pandas.pydata.org/docs/)
[Scikit-learn Documentation](https://scikit-learn.org/stable/)
[XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)

For further details, refer to the Big_Mart_Sales_Prediction.ipynb notebook.
