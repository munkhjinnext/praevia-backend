from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
from prophet import Prophet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/forecast")
async def forecast(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    if "Date" not in df.columns or "Revenue" not in df.columns:
        return {"error": "CSV must contain 'Date' and 'Revenue' columns"}

    # Prepare data for Prophet
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Date": "ds", "Revenue": "y"})
    df = df.sort_values("ds")

    # Fit model
    model = Prophet()
    model.fit(df)

    # Make future dataframe for 3 months
    future = model.make_future_dataframe(periods=3, freq='MS')
    forecast = model.predict(future)

    result = forecast[["ds", "yhat"]].tail(3)
    forecast_data = [
        {"date": row["ds"].strftime("%Y-%m-%d"), "forecasted_revenue": round(row["yhat"], 2)}
        for _, row in result.iterrows()
    ]

    return {"forecast": forecast_data}
