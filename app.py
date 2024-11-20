from shiny import App, ui, render
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64

def connect_to_database():
    # Replace with your MySQL credentials
    engine = create_engine("mysql+mysqlconnector://root:root@localhost/profile")
    return engine.connect()
# Fetch data from the database
def fetch_data():
    conn = connect_to_database()
    query = """
    SELECT h.dateOfVisit, 
           COUNT(br.id) AS population,
           SUM(CASE WHEN br.pregnant = 1 THEN 1 ELSE 0 END) AS pregnant,
           SUM(CASE WHEN br.lactating = 1 THEN 1 ELSE 0 END) AS lactating,
           AVG(CASE WHEN h.type_of_dwelling IS NOT NULL THEN 1 ELSE 0 END) AS type_of_dwelling,
           AVG(CASE WHEN h.type_of_toilet IS NOT NULL THEN 1 ELSE 0 END) AS type_of_toilet,
           AVG(CASE WHEN h.source_of_water IS NOT NULL THEN 1 ELSE 0 END) AS source_of_water,
           AVG(CASE WHEN h.garbage_disposal IS NOT NULL THEN 1 ELSE 0 END) AS garbage_disposal,
           AVG(CASE WHEN h.livestock > 0 THEN 1 ELSE 0 END) AS livestock,
           AVG(CASE WHEN h.using_iodize = 1 THEN 1 ELSE 0 END) AS using_iodize,
           AVG(CASE WHEN h.using_FP = 1 THEN 1 ELSE 0 END) AS using_FP,
           AVG(CASE WHEN h.member_of_4ps = 1 THEN 1 ELSE 0 END) AS member_of_4ps,
           AVG(CASE WHEN h.ip = 1 THEN 1 ELSE 0 END) AS ip,
           AVG(CASE WHEN h.uct = 1 THEN 1 ELSE 0 END) AS uct,
           AVG(CASE WHEN h.garden = 1 THEN 1 ELSE 0 END) AS garden
    FROM household h
    JOIN bns_resident br ON h.id = br.household_id
    WHERE h.status = 1 AND br.status = 1
    GROUP BY h.dateOfVisit
    ORDER BY h.dateOfVisit;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print(df.head())  # Debugging line to check data
    return df


# Define the UI
app_ui = ui.page_fluid(
    ui.h2("Population Growth Forecast and Diagnostics"),
    ui.row(
        ui.column(
            6,
            ui.input_action_button("forecast_button", "Generate Forecast and Diagnostics"),
            ui.br(),
            ui.h4("Forecasted Data"),
            ui.output_table("forecast_table")
        ),
        ui.column(
            6,
            ui.output_plot("forecast_plot", height="400px"),
        ),
    ),
    ui.br(),
    ui.h4("Diagnostics"),
    ui.output_text_verbatim("diagnostic_text"),
)

# Define the server logic
def server(input, output, session):
    @output
    @render.table
    def forecast_table():
        if input.forecast_button() == 0:
            return pd.DataFrame()
        
        data = fetch_data()
        data["dateOfVisit"] = pd.to_datetime(data["dateOfVisit"])
        monthly_data = data.groupby(data["dateOfVisit"].dt.to_period("M")).agg({
            "population": "sum",
            "pregnant": "mean",
            "lactating": "mean",
            "type_of_dwelling": "mean",
            "type_of_toilet": "mean",
            "source_of_water": "mean",
            "garbage_disposal": "mean",
            "livestock": "mean",
            "using_iodize": "mean",
            "using_FP": "mean",
            "member_of_4ps": "mean",
            "ip": "mean",
            "uct": "mean",
            "garden": "mean"
        }).reset_index()
        
        monthly_data["month"] = monthly_data["dateOfVisit"].dt.to_timestamp()
        population_ts = monthly_data["population"].values
        external_regressors = monthly_data.iloc[:, 2:].values

        # ARIMA Model
        model = ARIMA(population_ts, exog=external_regressors, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecasting
        forecast_horizon = 1
        forecast_result = model_fit.forecast(steps=forecast_horizon, exog=external_regressors[-forecast_horizon:])

        forecast_df = pd.DataFrame({
            "Month": pd.date_range(start="2024-12-01", periods=forecast_horizon, freq="M"),
            "Forecasted Population": forecast_result
        })
        
        return forecast_df

    @output
    @render.plot
    def forecast_plot():
        if input.forecast_button() == 0:
            return None
        
        data = fetch_data()
        data["dateOfVisit"] = pd.to_datetime(data["dateOfVisit"])
        monthly_data = data.groupby(data["dateOfVisit"].dt.to_period("M")).agg({
            "population": "sum"
        }).reset_index()
        monthly_data["month"] = monthly_data["dateOfVisit"].dt.to_timestamp()

        forecast_table_data = forecast_table()
        
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_data["month"], monthly_data["population"], label="Historical Population", color="blue")
        plt.plot(forecast_table_data["Month"], forecast_table_data["Forecasted Population"], label="Forecast", color="red")
        plt.xlabel("Month")
        plt.ylabel("Population")
        plt.title("Population Growth Forecast")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        return plt

    @output
    @render.text
    def diagnostic_text():
        if input.forecast_button() == 0:
            return "Click the button to generate diagnostics."
        
        data = fetch_data()
        if data.empty:
            return "No data available for diagnostics."

        data["dateOfVisit"] = pd.to_datetime(data["dateOfVisit"])
        monthly_data = data.groupby(data["dateOfVisit"].dt.to_period("M")).agg({
            "population": "sum"
        }).reset_index()
        population_ts = monthly_data["population"].values

        external_regressors = monthly_data.iloc[:, 2:].values
        model = ARIMA(population_ts, exog=external_regressors, order=(1, 1, 1))
        model_fit = model.fit()

        return str(model_fit.summary())


# Create the app
app = App(app_ui, server)
