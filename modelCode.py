##The main code to be used to train the model

#Import statements
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import json


##Read the CSV
dataset=pd.read_csv('data/train-test.csv',delimiter=',')

#drop load_id since it has no significance
dataset = dataset.drop(columns=["load_id"])

#covert date into numerical data for Catboost to understand
dataset["date"] = pd.to_datetime(dataset["date"])

dataset["year"] = dataset["date"].dt.year
dataset["month"] = dataset["date"].dt.month
dataset["day"] = dataset["date"].dt.day

dataset = dataset.drop(columns=["date"])

#Categorical features of the data
cat_features = ["pickup", "delivery", "equipment"]

# features and target [other columns, posted rate]
X = dataset.drop(columns=["posted_rate"])
y = dataset["posted_rate"]

#split data to train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#creating the validation set 
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)

print(dataset.info())

#model training
model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
    early_stopping_rounds=200,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val)
)

#Save the model 
model.save_model("posted_rate_model.cbm")

#Save the feature names
with open("feature_names.json", "w") as f:
    json.dump(list(X.columns), f)