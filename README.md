- Install the requirements using 'python -m pip install -r requirements.txt'
- to train the model run 'python modelCode.py' ensure the training data is accessible through 'data/train-test.csv'
- running creates 'posted_rate_model.cbm' - the model itself and 'feature_names.json' containing names of all the features used during training
- run  'python modelPredictor.py' to run the model and this create 'validation_predictions.csv' obtained from populating 'validation-predictions-template.csv'
- run 'python modelDecemberChartInputsPredictor.py' to populate 'december_chart_inputs.csv' 

  See code for detailed explanation on each line
