from catboost import CatBoostRegressor
import json
import pandas as pd

#load feature names
with open("feature_names.json") as f:
    feature_names = json.load(f)

model = CatBoostRegressor()
model.load_model("posted_rate_model.cbm")

# ##Read the CSV
validationDataset=pd.read_csv('data/validation.csv',delimiter=',')
validationPredictionDataset=pd.read_csv('data/validation-predictions-template.csv',delimiter=',')


#covert date into numerical data for Catboost to understand
validationDataset["date"] = pd.to_datetime(validationDataset["date"])

validationDataset["year"] = validationDataset["date"].dt.year
validationDataset["month"] = validationDataset["date"].dt.month
validationDataset["day"] = validationDataset["date"].dt.day

validationDataset = validationDataset.drop(columns=["date"])


for index, row in validationDataset.iterrows():
    predict_load = pd.DataFrame({
            "pickup": row["pickup"],
            "delivery": row["delivery"],
            "pickup_lat": row["pickup_lat"],
            "pickup_lon": row["pickup_lon"],
            "delivery_lat": row["delivery_lat"],
            "delivery_lon": row["delivery_lon"],
            "distance": row["distance"],
            "equipment": row["equipment"],
            "weight": row["weight"],
            "market_index": row["market_index"],
            "quote_signal": row["quote_signal"],
            "year": row["year"],
            "month": row["month"],
            "day": row["day"],
        },index=[0])
    prediction = model.predict(predict_load)
    validationPredictionDataset.loc[validationPredictionDataset["load_id"] == row["load_id"], "posted_rate"] = round(prediction[0], 2)
print("done saving changes")
validationPredictionDataset.to_csv("updated_validation-predictions-template.csv", index=False)