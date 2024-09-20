import pandas as pd
import numpy as np
from pmdarima import auto_arima
import re
from datetime import datetime
import gradio as gr
from calendar import monthrange

# Function to create sample data
def create_sample_data():
    """
    Creates a sample time series dataset with daily profit values, including seasonality and trend.
    Marks Sundays and official holidays in the 'control' column.
    """
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    profit = np.random.normal(loc=1000, scale=200, size=len(date_range))
    profit = np.abs(profit)  # Ensure profit is positive
    
    # Add seasonality
    profit += np.sin(np.arange(len(date_range)) * (2 * np.pi / 365)) * 200
    
    # Add trend
    profit += np.linspace(0, 500, len(date_range))
    
    df = pd.DataFrame({
        'date': date_range,
        'actualProfit': profit,
        'control': ''
    })
    
    # Mark Sundays
    df.loc[df['date'].dt.dayofweek == 6, 'control'] = 'Sunday'
    
    # Mark official holidays (e.g., 1st Jan and 4th July)
    for year in range(2020, 2024):
        df.loc[df['date'] == f"{year}-01-01", 'control'] = 'Official Holiday'
        df.loc[df['date'] == f"{year}-07-04", 'control'] = 'Official Holiday'
    
    return df

# Auto SARIMAX prediction function
def auto_sarimax_predict(month, num_working_days):
    """
    Predicts total profit for a given month and number of working days using an auto-configured SARIMAX model.
    """
    # Create sample data
    df = create_sample_data()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Create 'sunday' and 'official_holiday' binary indicators
    df['sunday'] = df['control'].apply(lambda x: 1 if 'Sunday' in x else 0)
    df['official_holiday'] = df['control'].apply(lambda x: 1 if 'Official Holiday' in x else 0)
    
    # Convert month to numerical value
    month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 
                  'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 
                  'November': 11, 'December': 12}
    month_num = month_dict[month]
    
    # Create month dummy variables
    df['month'] = df.index.month
    df = pd.get_dummies(df, columns=['month'], drop_first=True)
    
    # Exogenous variables
    exog_columns = ['sunday', 'official_holiday'] + [col for col in df.columns if col.startswith('month_')]
    exog = df[exog_columns]
    
    # Fit the auto_arima model
    model = auto_arima(df['actualProfit'], exogenous=exog, seasonal=True, m=12,
                       stepwise=True, suppress_warnings=True, D=1,
                       max_p=3, max_q=3, max_P=3, max_Q=3)
    
    # Generate future dates for the selected month in the next year
    prediction_year = df.index.year.max() + 1
    start_date = datetime(prediction_year, month_num, 1)
    num_days_in_month = monthrange(prediction_year, month_num)[1]
    date_range = pd.date_range(start=start_date, periods=num_days_in_month, freq='D')
    
    # Create future exogenous variables
    future_df = pd.DataFrame(index=date_range)
    # Create 'sunday' indicator
    future_df['sunday'] = future_df.index.dayofweek.apply(lambda x: 1 if x == 6 else 0)
    # Create 'official_holiday' indicator
    future_df['official_holiday'] = 0
    # Mark official holidays (assuming same dates)
    if datetime(prediction_year, 1, 1) in future_df.index:
        future_df.loc[datetime(prediction_year, 1, 1), 'official_holiday'] = 1
    if datetime(prediction_year, 7, 4) in future_df.index:
        future_df.loc[datetime(prediction_year, 7, 4), 'official_holiday'] = 1
    
    # Create month dummy variables
    future_df['month'] = future_df.index.month
    future_df = pd.get_dummies(future_df, columns=['month'], drop_first=True)
    
    # Ensure future_exog has the same columns as exog
    for col in exog_columns:
        if col not in future_df.columns:
            future_df[col] = 0
    future_exog = future_df[exog_columns]
    
    # Identify working days (excluding Sundays and official holidays)
    future_df['is_working_day'] = 1 - (future_df['sunday'] | future_df['official_holiday'])
    working_days_df = future_df[future_df['is_working_day'] == 1]
    working_days_exog = working_days_df[exog_columns]
    
    # Limit to the specified number of working days
    if len(working_days_exog) >= num_working_days:
        working_days_exog = working_days_exog.iloc[:int(num_working_days)]
    else:
        # Not enough working days in the month
        return f"Not enough working days in {month} {prediction_year} to match the specified number."
    
    # Make predictions
    predictions = model.predict(n_periods=len(working_days_exog), exogenous=working_days_exog)
    forecast_result = np.sum(predictions)
    
    return f"Selected month: {month}, Number of working days: {num_working_days}\nEstimated total profit: {forecast_result:.2f} USD"

# Create Gradio interface
interface = gr.Interface(
    fn=auto_sarimax_predict,
    inputs=[
        gr.Dropdown(
            choices=['January', 'February', 'March', 'April', 'May', 'June', 
                     'July', 'August', 'September', 'October', 'November', 'December'], 
            label="Select Month"
        ),
        gr.Number(
            label="Enter the number of working days in the selected month",
            value=20
        )
    ],
    outputs="text",
    title="Profit Forecast",
    description="Forecast total profit based on the selected month and number of working days."
)

if __name__ == "__main__":
    interface.launch()
