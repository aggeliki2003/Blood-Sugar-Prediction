import numpy as np
import pandas as pd
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
import os
from clark_utils import calculate_clarke_zones
from grid_2 import clarke_error_grid
from grid import clarke_error_grid_1
from cg_ega import CGEGA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
import tensorflow as tf
import random
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import optuna
import numpy as np
import pandas as pd
from scatter import scatter


# -----------------------------
# CONFIGURATION
# -----------------------------

DATA_PATH = '/Users/bioinfoadmin/Desktop/Aggeliki_code/Simulated_22'
DATA_PATH_2 = '/Users/bioinfoadmin/Desktop/Aggeliki_code/22_days_all'
DATA_PATH_3 = '/Users/bioinfoadmin/Desktop/Aggeliki_code/San_paper'

sim_files = [f for f in os.listdir(DATA_PATH) if f.endswith(('.csv'))]
ari_files = [a for a in os.listdir(DATA_PATH_2) if a.endswith(('.csv'))]
san_files = [s for s in os.listdir(DATA_PATH_3) if s.endswith(('.csv'))]

WINDOWS_MIN = [30, 60]
HORIZONS_MIN = [15, 30, 60]

# Create the sliding window
def create_sequences(data, window_minutes, ph_minutes, step = 5):
    w = window_minutes // step
    ph = ph_minutes // step
    x, y = [], []
    data = np.array(data)
    for i in range(w, len(data) - ph + 1):
        x.append(data[i - w:i])
        y.append(data[i + ph - 1, 0])

    x = np.array(x)
    y = np.array(y).reshape(-1, 1)

    return x, y

def add_summary_features(x_3d):
    # 1. Isolate just the CGM values across the time window (Feature index 0)
    cgm_window = x_3d[:, :, 0] 
    
    # 2. Calculate the statistics across the time window (axis=1)
    # keepdims=True ensures they stay formatted as neat vertical columns
    window_mean = np.mean(cgm_window, axis=1, keepdims=True)
    window_std  = np.std(cgm_window, axis=1, keepdims=True)
    window_min  = np.min(cgm_window, axis=1, keepdims=True)
    window_max  = np.max(cgm_window, axis=1, keepdims=True)
    
    # 3. Flatten the original 3D array so XGBoost can read it
    x_flattened = x_3d.reshape(x_3d.shape[0], -1)
    
    # 4. Glue the new statistical columns to the end of the flattened array
    x_enriched = np.hstack((x_flattened, window_mean, window_std, window_min, window_max))
    
    return x_enriched

# Build LSTM model / validation data will be split inside the function from x, y data
def lstm_built(x, y, x_test, y_test, name_sim, name_ari):

    #Reproducability
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Splitting data
    train_ratio = 0.8
    n = len(x)
    train_end = int(n * train_ratio)
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:], y[train_end:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], -1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], -1))

    # Give every sequence a base weight of 1.0
    train_weights = np.ones(len(y_train))

    # Multiply the importance of sequences where the target is low
    train_weights[y_train.flatten() < 100] = 5.0

    # Scaling data
    print(x_train.shape)
    n_samples_t, n_steps, n_features = x_train.shape
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train.reshape(-1, n_features)).reshape(n_samples_t, n_steps, n_features)
    x_val = scaler_x.transform(x_val.reshape(-1, n_features)).reshape(x_val.shape[0], n_steps, n_features)
    x_test = scaler_x.transform(x_test.reshape(-1, n_features)).reshape(x_test.shape[0], n_steps, n_features)

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))

    # Building model
    model = Sequential([
        LSTM(50, return_sequences = False, input_shape=(n_steps, n_features)),
        Dropout(0.2),
        Dense(8, activation = 'relu'),
        Dense(1)
    ])


    model.compile(optimizer='adam', loss='huber')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]



    model.fit(
        x_train,y_train,
        validation_data=(x_val, y_val),
        epochs=100,batch_size=128,
        shuffle=True,
        sample_weight=train_weights,
        callbacks = [callbacks])


    # Predict and inverse transform
    y_test_pred = model.predict(x_test, verbose=0)
    print(y_test_pred.shape)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    y_test = y_test

    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    zones = calculate_clarke_zones(y_test, y_test_pred)
    print(rmse, r2, mape, mse, mae)
    #scatter(y_test, y_test_pred)

    print(name_sim, name_ari)
    return {
        'Model': 'LSTM',
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'MSE': mse,
        'MAE': mae,
        'ZONE A' : zones['A'],
        'ZONE B' : zones['B'],
        'ZONE C' : zones['C'],
        'ZONE D' : zones['D'],
        'ZONE E' : zones['E'],
        'ZONE A+B': zones['A'] + zones['B'],
        'PRED' : y_test_pred,
        'TRUE' : y_test
    }

    #plot, zones = clarke_error_grid(y_test, y_test_pred, f"{file_name} LSTM Model Grid")
    #plot.show()

#-------------------------- Build Linear Regression Model ------------------------------------------------
def built_lr(x, y, x_test, y_test, name_sim, name_ari):

    model = LinearRegression()

    train_ratio = 0.8
    n = len(x)
    train_end = int(n * train_ratio)
    x_train, y_train = x[:train_end], y[:train_end]

    x_train = x_train.reshape(x_train.shape[0], -1) # Flattens timesteps and features
    x_test = x_test.reshape(x_test.shape[0], -1)

    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    # Metrics
    rmse_lr = np.sqrt(mean_squared_error(y_test, pred))
    r2_lr = r2_score(y_test, pred)
    mape_lr = mean_absolute_percentage_error(y_test, pred) * 100
    mse_lr = mean_squared_error(y_test, pred)
    mae_lr = mean_absolute_error(y_test, pred)
    zones = calculate_clarke_zones(y_test, pred)
    #plot, zones_lr = clarke_error_grid(y_test_lr, pred, "Linear Regression Real Data - Simulated Data \n")
    #plot.show()
    #scatter(y_test, pred)

    print(name_sim, name_ari)
    return {
        'Model': 'Linear Regression',
        'RMSE': rmse_lr,
        'R2': r2_lr,
        'MAPE': mape_lr,
        'MSE': mse_lr,
        'MAE': mae_lr,
        'ZONE A' : zones['A'],
        'ZONE B' : zones['B'],
        'ZONE C' : zones['C'],
        'ZONE D' : zones['D'],
        'ZONE E' : zones['E'],
        'ZONE A+B': zones['A'] + zones['B'],
        'PRED' : pred,
        'TRUE' : y_test
    }

#---------------------- Build Xgboost Model -------------------------


def built_xgboost(x, y, x_test, y_test, name_sim, name_ari):

    # Split data 
    train_ratio = 0.8
    n = len(x)
    train_end = int(n * train_ratio)

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:], y[train_end:]

    #x_train = add_summary_features(x_train)
    #x_val = add_summary_features(x_val)
    #x_test = add_summary_features(x_test)

    x_train = x_train.reshape(x_train.shape[0], -1) # Flattens timesteps and features
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)

    # Set optuna hyperparameter tuning
    def objective(trial):

        # Suggest hyperparameters
        params = {
                        "n_estimators": 1500,
                        "max_depth": trial.suggest_int("max_depth", 3, 11), # Reduced to prevent memorizing noise
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                        "min_child_weight": trial.suggest_int("min_child_weight", 2, 7),
                        
                        # --- NEW REGULARIZATION PARAMS ---
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),   # L1 Regularization (Lasso)
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True), # L2 Regularization (Ridge)
                        "gamma": trial.suggest_float("gamma", 0.1, 5.0),                       # Prunes useless splits
                        
                        "objective": "reg:squarederror",
                        "tree_method": "hist",
                        "random_state": 42,
                    }

        reg = xgb.XGBRegressor(
            **params,
            early_stopping_rounds = 50
        )
        reg.fit(x_train, y_train,
                eval_set = [(x_val, y_val)],
                verbose = 100)

        preds = reg.predict(x_val)
        rmse = root_mean_squared_error(y_val, preds)
        return rmse

    # Run optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)

    best_params = study.best_params

    final_model = xgb.XGBRegressor(
        **best_params,
        n_estimators=1100,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42
    )
    # 1. Start by giving every training sample a default weight of 1.0
    train_weights = np.ones(len(y_train))
    val_weights = np.ones(len(y_val))

    # 2. Identify the rare/low values and multiply their importance (e.g., by 5)
    # You can tweak this multiplier (5.0) up or down based on your results
    train_weights[y_train.flatten() < 100] = 5.0
    val_weights[y_val.flatten() < 100] = 5.0

    print(f"Number of boosted training samples: {np.sum(y_train < 100)}")

    final_model.fit(
        x_train, y_train,
        sample_weight=train_weights,          
        eval_set=[(x_val, y_val)],
        sample_weight_eval_set=[val_weights],
        verbose=100
    )

    xg_pred = final_model.predict(x_test)

    # Metrics
    rmse_xg = np.sqrt(mean_squared_error(y_test, xg_pred))
    r2_xg = r2_score(y_test, xg_pred)
    mape_xg = mean_absolute_percentage_error(y_test, xg_pred) * 100
    mse_xg = mean_squared_error(y_test,xg_pred)
    mae_xg = mean_absolute_error(y_test, xg_pred)
    zones = calculate_clarke_zones(y_test, xg_pred)
    #scatter(y_test, xg_pred)
    print(rmse_xg, r2_xg, mape_xg, mse_xg, mae_xg)
            
    #plot, zones_xg = clarke_error_grid(y_test, xg_pred, "XGBoost  Real Data - Simulated Data \n ")
    #plot.show()
    print(name_sim, name_ari)
    return {
        'Model': 'XGBoost',
        'RMSE': rmse_xg,
        'R2': r2_xg,
        'MAPE': mape_xg,
        'MSE': mse_xg,
        'MAE': mae_xg,
        'ZONE A' : zones['A'],
        'ZONE B' : zones['B'],
        'ZONE C' : zones['C'],
        'ZONE D' : zones['D'],
        'ZONE E' : zones['E'],
        'ZONE A+B': zones['A'] + zones['B'],
        'PRED' : xg_pred,
        'TRUE' : y_test
    }
   


# Run all the models 
all_results = []

for name_sim in sim_files:
    for name_ari in san_files:
        #for name_san in san_files:

        sim = pd.read_csv(os.path.join(DATA_PATH, name_sim), parse_dates=['DateTime'])
        #ari = pd.read_csv(os.path.join(DATA_PATH_2, name_ari), parse_dates=['EventDateTime'])
        san = pd.read_csv(os.path.join(DATA_PATH_3, name_ari), parse_dates=['Date'])

        for w in WINDOWS_MIN:
            for ph in HORIZONS_MIN:

                # Let's say 'raw_cgm_values' is your chronological array of ~7000 glucose readings
                df_1 = pd.DataFrame(sim['CGM'])
                df = pd.DataFrame(san['CGM (mg / dl)'])

                # 1. Calculate the Rate of Change (RoC)
                # .diff() subtracts the previous row from the current row
                df['RoC'] = df['CGM (mg / dl)'].diff()
                df_1['RoC'] = df_1['CGM'].diff()

                # 2. Drop the first row (since the very first reading has no previous reading to subtract)
                df = df.dropna().reset_index(drop=True)
                df_1 = df_1.dropna().reset_index(drop=True)

                # 3. Convert back to a NumPy array to feed into your sequence generator
                # Now, instead of shape (6999, 1), your data is shape (6999, 2)
                sim_f = df.values 
                ari_f = df_1.values 

                print(df.head())
                print(sim_f.shape)
                print(ari_f.shape)

                x_sim, y_sim = create_sequences(sim_f, w, ph)
                x_ari, y_ari = create_sequences(ari_f, w, ph)
                #x_san, y_san = create_sequences(san, w, ph)

                # Run lstm
                results_lstm = lstm_built(x_ari, y_ari, x_sim, y_sim, name_sim, name_ari)
                results_lstm.update({'Window': w, 'Horizon': ph, 'Sim_File': name_sim, 'Ari_File': name_ari})
                all_results.append(results_lstm)
                # Run lr
                results_lr = built_lr( x_ari, y_ari, x_sim, y_sim, name_sim, name_ari)
                results_lr.update({'Window': w, 'Horizon': ph, 'Sim_File': name_sim, 'Ari_File': name_ari})
                all_results.append(results_lr)
                # Run xgboost
                results_xg = built_xgboost( x_ari, y_ari, x_sim, y_sim, name_sim, name_ari)
                results_xg.update({'Window': w, 'Horizon': ph, 'Sim_File': name_sim, 'Ari_File': name_ari})
                all_results.append(results_xg)

# Convert the master list of dictionaries into a Pandas DataFrame
df_results = pd.DataFrame(all_results)

# Group by Model, Window, and Horizon, then calculate mean and standard deviation
# 1. Define which columns are single numbers (to be averaged)
stats_cols = ['RMSE', 'R2', 'MAPE', 'MSE', 'MAE', 'ZONE A', 'ZONE B', 'ZONE C', 'ZONE D', 'ZONE E', 'ZONE A+B']

# 2. Create a dictionary for the aggregation
# For stats: calculate mean and std
# For arrays: just "take" them (or use 'first' / list)
agg_dict = {col: ['mean', 'std'] for col in stats_cols}

# Add the array columns - we just want to keep them as they are
# Note: 'first' works if the arrays are identical within the group. 
# If they are different, you might want 'sum' or to wrap them in a list.
agg_dict['PRED'] = lambda x: list(x) 
agg_dict['TRUE'] = lambda x: list(x)

# 3. Run the groupby
summary_stats = df_results.groupby(['Model', 'Window', 'Horizon']).agg(agg_dict).reset_index()

# 4. Flatten the multi-level column names
# This handles the lambda columns specifically so they don't end up named "PRED_<lambda>"
new_cols = []
for col in summary_stats.columns.values:
    if col[1] == 'mean' or col[1] == 'std':
        new_cols.append(f"{col[0]}_{col[1]}")
    elif col[0] in ['PRED', 'TRUE']:
        new_cols.append(col[0]) # Keep 'PRED' and 'TRUE' clean
    else:
        new_cols.append(col[0])
summary_stats.columns = new_cols


# Export both the raw results and the summary stats to a single Excel file
output_path = '/Users/bioinfoadmin/Desktop/Aggeliki_code/Sim_Train_San_paper_model_50.xlsx'

# Create a copy for exporting so we don't mess up your actual data in memory
export_df = summary_stats.copy()

# Convert the list/array columns into strings so Excel can handle them
if 'PRED' in export_df.columns:
    export_df['PRED'] = export_df['PRED'].astype(str)
if 'TRUE' in export_df.columns:
    export_df['TRUE'] = export_df['TRUE'].astype(str)

with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    df_results.to_excel(writer, sheet_name='Raw_Data', index=False)
    # Use the export version here
    export_df.to_excel(writer, sheet_name='Averages_and_Stdev', index=False)

print(f"Success! Results saved to {output_path}")

# -------------------------------------------------------------------
# NEW: GENERATE AGGREGATED CLARKE ERROR GRIDS
# -------------------------------------------------------------------
# 1. Define the directory and create the folder ONCE before the loop

save_dir = r"C:\Users\bioinfoadmin\Desktop\Aggeliki_code\Plots_sim_train_san_paper_model_50"
os.makedirs(save_dir, exist_ok=True)

# Loop through every unique combination of Model, Window, and Horizon
for index, row in summary_stats.iterrows():
    model_name = row['Model']
    window = row['Window']
    horizon = row['Horizon']
    
    # row['PRED'] and row['TRUE'] are currently lists of arrays (one array per subject)
    # np.concatenate() glues them all together. .flatten() ensures they are simple 1D lists.
    all_preds_aggregated = np.concatenate(row['PRED']).flatten()
    all_trues_aggregated = np.concatenate(row['TRUE']).flatten()
    
    # Create a clean title for the plot
    grid_title = f"{model_name} - All Subjects\nWindow: {window}m | Horizon: {horizon}m"
    
    # Call your existing clarke_error_grid function
    # (Make sure clarke_error_grid returns the plot object so you can show/save it)
    plot, aggregated_zones = clarke_error_grid(all_trues_aggregated, all_preds_aggregated, grid_title)
    
    # Build the safe, OS-friendly file path
    file_name = os.path.join(save_dir, f"Clarke_{model_name}_W{window}_H{horizon}.png")
    plot.savefig(file_name, bbox_inches='tight')
    plot.close() # Close to free up memory
# -------------------------------------------------------------------

