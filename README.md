# ProfitPredictor: Forecasting Future Profits with Auto SARIMAX Model

This project demonstrates how to forecast total profit for a selected month and number of working days using an auto-configured SARIMAX model. The model accounts for seasonality, trends, and exogenous variables such as weekends and official holidays. An interactive user interface is built using Gradio to allow users to input parameters and receive forecasts in real-time.

## Overview

The **ProfitPredictor** project involves:

- Generating a synthetic time series dataset simulating daily profit over several years.
- Marking special days such as Sundays and official holidays.
- Training an auto-configured SARIMAX model using the historical data.
- Forecasting future profits based on user-selected month and working days.
- Providing an interactive web interface for user interaction.

## Features

- **Time Series Forecasting**: Utilizes the SARIMAX model to predict future profits.
- **Seasonality and Trend**: Incorporates seasonality and trend in the data generation and modeling process.
- **Exogenous Variables**: Considers weekends and official holidays as exogenous variables.
- **Interactive Interface**: Employs Gradio to create a user-friendly web interface for making predictions.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ProfitPredictor.git
   cd ProfitPredictor
