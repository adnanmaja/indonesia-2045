import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

datadir = "D:/Python/2045/data/2045-sets-v10.csv"

# Data cleaning
def dataclean(df_raw):
    Tahun = ['2010 IPM', '2011 IPM', '2012 IPM', '2013 IPM', '2014 IPM', '2015 IPM', '2016 IPM', '2017 IPM', '2018 IPM', '2019 IPM', '2020 IPM',
            '2010 Kemiskinan', '2011 Kemiskinan', '2012 Kemiskinan', '2013 Kemiskinan']
    for i in Tahun:
        df_raw[i] = df_raw[i].astype(str)
        df_raw[i] = df_raw[i].str.replace(',', '.')
        df_raw[i] = df_raw[i].str.replace(' ', '')  
        df_raw[i] = df_raw[i].replace(['', 'nan', 'None'], np.nan)  
        df_raw[i] = pd.to_numeric(df_raw[i], errors='coerce')  

    df_melt = df_raw.melt(id_vars=['Provinsi'], 
                            var_name='Tahun_Indikator', 
                            value_name='Value')

    df_melt[['Tahun', 'Indikator']] = df_melt['Tahun_Indikator'].str.split(' ', n=1, expand=True)
    df_melt = df_melt.drop('Tahun_Indikator', axis=1)
    df_melt = df_melt[['Provinsi', 'Tahun', 'Indikator', 'Value']]
    df_melted = df_melt.pivot(index=['Provinsi', 'Tahun'], 
                            columns='Indikator', 
                            values='Value').reset_index()
    df_melted.columns.name = None

    df_melted.to_csv('cleaned-data.csv')
    df_melted.to_json('cleaned-data.json')
    return df_melted


df_melted = dataclean(pd.read_csv(datadir))


'''

--- PENGANGGURAN PIPELINE ---

'''

# Getting the data ready for pengangguran
def pengangguran_setup(df_melted):
    df_melted = df_melted.dropna(subset=['Inflasi'])
    df_melted = df_melted.dropna(subset=['Upah Minimum'])
    X = df_melted[['IPM', 'Inflasi', 'PDRB_ADHB', 'PDRB_ADHK', 'PDRBK_ADHB', 'PDRBK_ADHK', 'Upah Minimum', 'Kemiskinan']]
    y = df_melted['Pengangguran']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Random Forest on Pengangguran 
def pengangguran_rf():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_melted)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, y_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rf_r2 = r2_score(y_test, y_pred)

    return rf_mae, rf_rmse, rf_r2

# Linear Regression on Pengangguran
def pengangguran_lr():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_melted)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lr_r2 = r2_score(y_test, y_pred)
    return lr_r2

# Grid Search on Pengangguran
def pengangguran_grid():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_melted)
    param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
    grid.fit(X_train, y_train)

    grid_r2 = grid.best_score_
    grid_bestparams = grid.best_params_

    return grid_r2, grid_bestparams

# XGB Regressor on Pengangguran
def pengangguran_xgb():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_melted)
    model = xgb.XGBRegressor(
        objective="reg:linear",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    xgb_mse = mean_squared_error(y_test, y_pred)
    xgb_r2 = r2_score(y_test, y_pred)
    return xgb_mse, xgb_r2

# LGBM Regressor on Pengangguran
def pengangguran_lgb():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_melted)
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    lgb_r2 = r2_score(y_test, y_pred)
    return lgb_r2, y_pred

# Comparing results form all the models tested on pengangguran
def pengangguran_results():
    rf_mae, rf_rmse, rf_r2 = pengangguran_rf()
    lr_r2 = pengangguran_lr()
    grid_r2, grid_bestparams = pengangguran_grid() 
    xgb_mse, xgb_r2 = pengangguran_xgb()
    lgb_r2 = pengangguran_lgb()
    print(f"Random Forest | r2: {rf_r2} | mae: {rf_mae} | rmse: {rf_rmse}")
    print(f"Grid Search CV| r2: {grid_r2} (best) | params: {grid_bestparams} (best)")
    print(f"Linear Regression | r2: {lr_r2}")
    print(f"XGB Regressor | r2: {xgb_r2} | mse: {xgb_mse}")
    print(f"LGBM Regressor | r2: {lgb_r2}")

# Visualizing actual vs predicted results from the LGBM Regression model
def pengangguran_lgbm_visualization():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_melted)
    lgb_r2, y_pred = pengangguran_lgb()
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'LGBM: Actual vs Predicted (R² = {lgb_r2:.3f})')
    plt.show()  


pengangguran_results()




#Kemiskinan pipeline
def kemiskinan_pipe():
    X = df_melted[['IPM', 'Inflasi', 'PDRB_ADHB', 'PDRB_ADHK', 'PDRBK_ADHB', 'PDRBK_ADHK', 'Upah Minimum', 'Pengangguran']]
    y = np.log(df_melted['Kemiskinan'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    actual_y_pred = np.exp(y_pred)
    actual_y_test = np.exp(y_test)

    mae = mean_absolute_error(actual_y_test, actual_y_pred)
    rmse = np.sqrt(mean_squared_error(actual_y_test, actual_y_pred))
    r2 = r2_score(actual_y_test, actual_y_pred)


    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R² Score:", r2)

    print(actual_y_pred)


    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)

