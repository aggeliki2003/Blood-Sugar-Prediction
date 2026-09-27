# Blood Sugar Prediction: Comparative Modeling of Glucose Dynamics

This repository contains the code and implementation details for the paper: **"Comparative Modeling of Glucose Dynamics in Patients with Type 1 Diabetes based on Simulated and Real Data"** by Angeliki Panta and Costas Papaloukas.

## Overview
This project evaluates the effectiveness of three different machine learning models—Linear Regression (LR), Long Short-Term Memory (LSTM) networks, and eXtreme Gradient Boosting (XGBoost)—in predicting future continuous glucose monitoring (CGM) values. Predictions are made for 15, 30, and 60-minute horizons using 30 or 60 minutes of historical CGM data. 

The models are trained and cross-evaluated on a mix of simulated data (*in silico*) and real-world clinical datasets to assess prediction accuracy, generalizability, and clinical safety using Clarke Error Grid Analysis (CEGA).

## Dependencies
The code is developed in **Python 3.10**. To run the models, you will need the following primary libraries:
* `tensorflow` (v2.20) - For the LSTM model
* `xgboost` (v3.2.0) - For the XGBoost model
* `scikit-learn` (v1.7.2) - For Linear Regression, MinMaxScaler, and metrics
* `optuna` - For XGBoost hyperparameter tuning
* `pandas` & `numpy` - For data manipulation and preprocessing

## Datasets
The project utilizes three distinct datasets. Due to privacy and licensing, please obtain the datasets from their original sources:

1. **In Silico Dataset:** Generated using the UVa-Padova simulator.
   * Simulator implementation available at: [simglucose](https://github.com/jxx123/simglucose)
2. **AZT1D Dataset:** Real-world CGM data from 25 patients.
   * Available at: [Mendeley Data](https://doi.org/10.17632/gk9m674wcx.1)
3. **ShanghaiT1DM Dataset:** Independent real-world dataset from 12 patients.
   * Available at: [Figshare](https://doi.org/10.6084/m9.figshare.20444397)

## Model Architectures
* **Linear Regression:** Serves as the baseline model using a sliding window of historical glucose data.
* **LSTM:** A deep learning sequential model configured with 50 units, a dropout rate of 0.2, and two dense layers (8 and 1 units). Uses the Adam optimizer and Huber loss.
* **XGBoost:** A tree-based ensemble model leveraging gradient descent. Hyperparameters (such as depth, learning rate, and tree splitting thresholds) were optimized using the Optuna framework.

## Evaluation Metrics
Model performance is quantified using:
1. **Root Mean Squared Error (RMSE):** To measure numerical prediction error.
2. **Coefficient of Determination ($R^2$):** To measure the variance explained by the model.
3. **Clarke Error Grid Analysis (CEGA):** To evaluate the clinical safety and decision-making impact of the predictions (focusing on the clinically acceptable Zones A and B).

## Authors & Citation
**Angeliki Panta** and **Costas Papaloukas**  
Department of Biological Applications and Technology, University of Ioannina, Greece.

If you use this code or methodology in your research, please cite the associated MDPI paper.