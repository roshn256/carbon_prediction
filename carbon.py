# carbon_prediction_final.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import os
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima  # pip install pmdarima

def load_data():
    """Load and preprocess the dataset from Excel"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_files = [
            os.path.join(current_dir, "carbon_data.xls"),
            os.path.join(current_dir, "carbon_data.xlsx"),
            os.path.join(current_dir, "data", "carbon_data.xls"),
            os.path.join(current_dir, "data", "carbon_data.xlsx")
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                print(f"Found file at: {file_path}")
                df = pd.read_excel(file_path)
                break
        else:
            raise FileNotFoundError("Could not find carbon_data.xls or carbon_data.xlsx")
        
        # Melt and clean data
        df_long = pd.melt(
            df,
            id_vars=['Country code', 'Country name', 'Series code', 'Series name', 'SCALE', 'Decimals'],
            var_name='Year',
            value_name='Value'
        )
        df_long['Year'] = df_long['Year'].astype(int)
        df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
        
        # Filter for CO2-related metrics
        co2_series_codes = ['EN.ATM.CO2E.KT', 'EN.ATM.CO2E.PC']
        df_co2 = df_long[df_long['Series code'].isin(co2_series_codes)].dropna()
        
        # Pivot and clean column names
        df_pivot = df_co2.pivot_table(
            index=['Country code', 'Country name', 'Year'],
            columns='Series name',
            values='Value'
        ).reset_index()
        df_pivot.columns = [str(col).strip() for col in df_pivot.columns]
        
        return df_pivot
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def check_stationarity(series):
    """Check if time series is stationary"""
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    return result[1] < 0.05

def arima_forecast(df, country_code='USA'):
    """Improved ARIMA forecasting with proper time handling"""
    if df is None:
        return
        
    co2_col = [col for col in df.columns if 'CO2' in col or 'co2' in col]
    if not co2_col:
        print("No CO2-related column found")
        return
    co2_col = co2_col[0]
    
    print(f"\n=== ARIMA Forecast for {country_code} ===")
    country_data = df[df['Country code'] == country_code]
    
    if country_data.empty:
        print(f"No data for {country_code}")
        return
    
    # Prepare time series with proper frequency
    ts_data = country_data.set_index('Year')[co2_col]
    ts_data.index = pd.to_datetime(ts_data.index, format='%Y')
    ts_data = ts_data.asfreq('YE')  # Year-End frequency
    
    # Check stationarity
    print("\nStationarity Check:")
    if not check_stationarity(ts_data):
        print("Data is not stationary - differencing will be used")
    
    # Auto-select best ARIMA parameters
    print("\nFinding best ARIMA parameters...")
    model = auto_arima(ts_data, seasonal=False, trace=True)
    print(f"Best ARIMA order: {model.order}")
    
    # Fit model with best parameters
    arima_model = ARIMA(ts_data, order=model.order)
    model_fit = arima_model.fit()
    
    # Forecast next 5 years
    forecast = model_fit.forecast(steps=5)
    last_year = ts_data.index.year[-1]
    
    print("\nForecast for USA (next 5 years):")
    for year, value in zip(range(last_year+1, last_year+6), forecast):
        print(f"{year}: {value:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(ts_data.index, ts_data, label='Historical')
    forecast_index = pd.date_range(
        start=f"{last_year+1}-12-31",
        periods=5,
        freq='YE'
    )
    plt.plot(forecast_index, forecast, 'r--', label='Forecast')
    plt.title(f"{co2_col} Forecast for {country_code}")
    plt.xlabel("Year")
    plt.ylabel(co2_col)
    plt.legend()
    plt.tight_layout()
    plt.show()

def machine_learning_model(df):
    """Random Forest model with time-series features"""
    if df is None:
        return None
        
    print("\n=== Machine Learning Model ===")
    co2_col = [col for col in df.columns if 'CO2' in col or 'co2' in col]
    if not co2_col:
        print("No CO2-related column found")
        return None
    co2_col = co2_col[0]
    
    # Feature engineering
    df_ml = df.copy()
    for lag in [1, 2, 3]:
        df_ml[f'CO2_lag_{lag}'] = df_ml.groupby('Country code')[co2_col].shift(lag)
    df_ml = df_ml.dropna()
    
    # Prepare data
    X = df_ml[['Year'] + [col for col in df_ml.columns if 'CO2_lag_' in col]]
    y = df_ml[co2_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train and evaluate
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    
    return rf_model

def main():
    """Main execution function"""
    print("Starting Carbon Prediction Analysis...")
    df = load_data()
    
    if df is not None:
        # EDA
        exploratory_analysis(df)
        
        # ARIMA Forecasting
        arima_forecast(df, 'USA')
        
        # Machine Learning
        ml_model = machine_learning_model(df)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
