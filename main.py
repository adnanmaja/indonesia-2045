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

# Feature engineering
def growth(df_melted):
    df = df_melted.copy()

    df['PDRB_ADHK_YoY'] = df.groupby('Provinsi')['PDRB_ADHK'].pct_change(fill_method=None) * 100
    df['PDRB_ADHB_YoY'] = df.groupby('Provinsi')['PDRB_ADHB'].pct_change(fill_method=None) * 100
    df['PDRBK_ADHK_YoY'] = df.groupby('Provinsi')['PDRBK_ADHK'].pct_change(fill_method=None) * 100
    df['PDRBK_ADHB_YoY'] = df.groupby('Provinsi')['PDRBK_ADHB'].pct_change(fill_method=None) * 100

    pivot_adhk = df.pivot(index='Provinsi', columns='Tahun', values='PDRB_ADHK')
    pivot_adhb = df.pivot(index='Provinsi', columns='Tahun', values='PDRB_ADHB')
    pivot_bk_adhk = df.pivot(index='Provinsi', columns='Tahun', values='PDRBK_ADHK')
    pivot_bk_adhb = df.pivot(index='Provinsi', columns='Tahun', values='PDRBK_ADHB')

    common_provinces = pivot_adhk.dropna(subset=['2010', '2020']).index

    pct_adhk = ((pivot_adhk.loc[common_provinces, '2020'] - pivot_adhk.loc[common_provinces, '2010']) / pivot_adhk.loc[common_provinces, '2010']) * 100
    pct_adhb = ((pivot_adhb.loc[common_provinces, '2020'] - pivot_adhb.loc[common_provinces, '2010']) / pivot_adhb.loc[common_provinces, '2010']) * 100
    pct_bk_adhk = ((pivot_bk_adhk.loc[common_provinces, '2020'] - pivot_bk_adhk.loc[common_provinces, '2010']) / pivot_bk_adhk.loc[common_provinces, '2010']) * 100
    pct_bk_adhb = ((pivot_bk_adhb.loc[common_provinces, '2020'] - pivot_bk_adhb.loc[common_provinces, '2010']) / pivot_bk_adhb.loc[common_provinces, '2010']) * 100

    df['PDRB_ADHK_Pct'] = df['Provinsi'].map(pct_adhk)
    df['PDRB_ADHB_Pct'] = df['Provinsi'].map(pct_adhb)
    df['PDRBK_ADHK_Pct'] = df['Provinsi'].map(pct_bk_adhk)
    df['PDRBK_ADHB_Pct'] = df['Provinsi'].map(pct_bk_adhb)

    return df
    
df_final = growth(df_melted)



# Getting the data ready for pengangguran
def pengangguran_setup(df_final):
    df_final = growth(df_melted)
    df_final = df_final.dropna()
    X = df_final[['IPM', 'Inflasi', 'PDRB_ADHB', 'PDRB_ADHK', 'PDRBK_ADHB', 'PDRBK_ADHK', 'Upah Minimum', 'Kemiskinan',
        'PDRB_ADHK_Pct', 'PDRB_ADHB_Pct', 'PDRB_ADHK_YoY', 'PDRB_ADHB_YoY', 'PDRBK_ADHK_Pct', 'PDRBK_ADHB_Pct', 'PDRBK_ADHK_YoY', 'PDRBK_ADHB_YoY']]
    y = df_final['Pengangguran']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Random Forest on Pengangguran 
def pengangguran_rf():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_final)

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
    X_train, X_test, y_train, y_test = pengangguran_setup(df_final)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lr_r2 = r2_score(y_test, y_pred)
    return lr_r2

# Grid Search on Pengangguran
def pengangguran_grid():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_final)
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
    X_train, X_test, y_train, y_test = pengangguran_setup(df_final)
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
    X_train, X_test, y_train, y_test = pengangguran_setup(df_final)
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    lgb_r2 = r2_score(y_test, y_pred)
    return lgb_r2, model

# Comparing results form all the models tested on pengangguran
def pengangguran_results():
    rf_mae, rf_rmse, rf_r2 = pengangguran_rf()
    lr_r2 = pengangguran_lr()
    grid_r2, grid_bestparams = pengangguran_grid() 
    xgb_mse, xgb_r2 = pengangguran_xgb()
    lgb_r2 = pengangguran_lgb()
    print(f"Random Forest | r2: {rf_r2} | mae: {rf_mae} | rmse: {rf_rmse}") # r2: 0.7316760837590445
    print(f"Grid Search CV| r2: {grid_r2} (best) | params: {grid_bestparams} (best)") # r2: 0.6175768683969117
    print(f"Linear Regression | r2: {lr_r2}") # r2: 0.0004658497515611648
    print(f"XGB Regressor | r2: {xgb_r2} | mse: {xgb_mse}") # r2: 0.7226360516283026
    print(f"LGBM Regressor | r2: {lgb_r2}") # r2: 0.7378041160824718

# Visualizing actual vs predicted results from the LGBM Regression model
def pengangguran_lgbm_visualization():
    X_train, X_test, y_train, y_test = pengangguran_setup(df_final)
    lgb_r2, y_pred = pengangguran_lgb()
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'LGBM: Actual vs Predicted (R² = {lgb_r2:.3f})')
    plt.show()  

# Other visulization types
def corr_heatmap():
    df = df_final.select_dtypes(include='number')
    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Correlation Heatmap of Economic Indicators')
    plt.tight_layout()
    plt.show()

def box_plot():
    sns.boxplot(data=df_final, x='Provinsi', y='Upah Minimum', hue='Provinsi')  
    plt.title('Perbandingan UMR tiap provinsi')
    plt.xlabel('Provinsi')
    plt.ylabel('UMR')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def violin_plot():
    sns.violinplot(data=df_final, x='Provinsi', y='IPM') 
    plt.title('Distribusi IPM tiap provinsi')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def lm_plot():
    sns.lmplot(data=df_final, x='Kemiskinan', y='Pengangguran', hue='Provinsi', scatter_kws={'alpha': 0.6}, 
        palette='tab10', ci=68)
    plt.show()


def importance():
    X = df_final[['IPM', 'Inflasi', 'PDRB_ADHB', 'PDRB_ADHK', 'PDRBK_ADHB', 'PDRBK_ADHK', 'Upah Minimum', 'Kemiskinan',
        'PDRB_ADHK_Pct', 'PDRB_ADHB_Pct', 'PDRB_ADHK_YoY', 'PDRB_ADHB_YoY', 'PDRBK_ADHK_Pct', 'PDRBK_ADHB_Pct', 'PDRBK_ADHK_YoY', 'PDRBK_ADHB_YoY']]
    _, model = pengangguran_lgb()
    importances = model.feature_importances_
    features = X.columns

    imp_series = pd.Series(importances, index=features)
    imp_series = imp_series.sort_values(ascending=True)

    plt.figure(figsize=(12, 6))
    plt.barh(features, imp_series.values)
    plt.title("Faktor Paling Berpengaruh Dalam Pengangguran")
    plt.xlabel("Feature Importance")
    plt.show()


def hypotetical_data():
    optimisticdir = "D:/Python/2045/data/Optimistic-editedv2.csv"
    stagnantdir = "D:/Python/2045/data/Stagnant-editedv2.csv"

    df_opt = pd.read_csv(optimisticdir)
    df_stag = pd.read_csv(stagnantdir)

    return df_opt, df_stag


def optimistic_pred():
    optimistic_scenario_data, df_stag = hypotetical_data() 
    lg_r2, trained_model = pengangguran_lgb()

    tahuns = [2026, 2027, 2028, 2029, 2030]
    op_pred = {}

    for tahun in tahuns:
        year_data = optimistic_scenario_data[
            (optimistic_scenario_data['Tahun'] == tahun) & 
            (optimistic_scenario_data['Provinsi'] == 'DI Yogyakarta')]
    
        features = year_data.drop(['Provinsi', 'Tahun'], axis=1)
         
        pred = trained_model.predict(features)
        op_pred[tahun] = pred[0]
        print(f"Tahun {tahun} | Optimis: {pred[0]:.2f}%")

    return op_pred


def stagnant_pred():
    df_opt, stagnant_data = hypotetical_data() 
    lg_r2, trained_model = pengangguran_lgb()

    tahuns = [2026, 2027, 2028, 2029, 2030]
    st_pred = {}

    for tahun in tahuns :
        year_data = stagnant_data[
            (stagnant_data['Tahun'] == tahun) & 
            (stagnant_data['Provinsi'] == 'DI Yogyakarta')]
    
        features = year_data.drop(['Provinsi', 'Tahun'], axis=1)
        
        pred = trained_model.predict(features)
        st_pred[tahun] = pred[0]
        print(f"Tahun {tahun} | Stagnan: {pred[0]:.2f}%")

    return st_pred

def pred_results():
    optimistic_pred()
    stagnant_pred()
pred_results()