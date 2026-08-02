from catboost import CatBoostRegressor
import json
import pandas as pd

#load feature names
with open("feature_names.json") as f:
    feature_names = json.load(f)

model = CatBoostRegressor()
model.load_model("posted_rate_model.cbm")

#Since we only have pickup,delivery,distance,equipment,weight,date,predicted_rate 
#We generate the missing features pickup_lat,pickup_lon,delivery_lat,delivery_lon,market_index,quote_signal 

# we can use this as constants since they are locations
#Using the training set we get a pattern 36.99152,-84.99876,41.31561,-85.36206 as pickup_lat,pickup_lon,delivery_lat,delivery_lon,
pickup_lat=36.99152
pickup_lon=-84.99876
delivery_lat=41.31561
delivery_lon=-85.36206

#leaves us with market_index,quote_signal Which we get the average with Lexington as pickup,Fort Wayne as delivery and with Dry van as equipment i.e. 21 entries
avgMarketIndex=1.047568571
avgQuoteSignal=2.023341905

# ##Read the CSV
decemberChartInputs=pd.read_csv('data/december-chart-inputs.csv',delimiter=',')


#covert date into numerical data for Catboost to understand
decemberChartInputs["date"] = pd.to_datetime(decemberChartInputs["date"])

decemberChartInputs["year"] = decemberChartInputs["date"].dt.year
decemberChartInputs["month"] = decemberChartInputs["date"].dt.month
decemberChartInputs["day"] = decemberChartInputs["date"].dt.day



for index, row in decemberChartInputs.iterrows():
    predict_load = pd.DataFrame({
            "pickup": row["pickup"],
            "delivery": row["delivery"],
            "pickup_lat":pickup_lat,
            "pickup_lon": pickup_lon,
            "delivery_lat": delivery_lat,
            "delivery_lon": delivery_lon,
            "distance": row["distance"],
            "equipment": row["equipment"],
            "weight": row["weight"],
            "market_index": avgMarketIndex,
            "quote_signal": avgQuoteSignal,
            "year": row["year"],
            "month": row["month"],
            "day": row["day"],
        },index=[0])
    prediction = model.predict(predict_load)
    decemberChartInputs.loc[index, "predicted_rate"] = round(prediction[0], 2)
print("done saving changes")

#Drop generated columns
decemberChartInputs = decemberChartInputs.drop(columns=["day"])
decemberChartInputs = decemberChartInputs.drop(columns=["month"])
decemberChartInputs = decemberChartInputs.drop(columns=["year"])

#Back to CSV
decemberChartInputs.to_csv("predictions_december_chart_inputs.csv", index=False)