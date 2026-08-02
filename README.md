1. Install the requirements using 'python -m pip install -r requirements.txt'
### PREREQUISITE
**IMPORTANT: Please review the file paths in the following lines of code before running**
+ modelCode.py :
  + line 11 - dataset=pd.read_csv('data/train-test.csv',delimiter=',')
+ modelPredictor.py :
  + line 13 - dataset=pd.read_csv('data/train-test.csv',delimiter=',')
  + line 14 - validationPredictionDataset=pd.read_csv('data/validation-predictions-template.csv',delimiter=',')
  + line 47 - validationPredictionDataset.to_csv("validation_predictions.csv", index=False)
+ modelDecemberChartInputsPredictor.py :
  + line 27 - decemberChartInputs=pd.read_csv('data/december-chart-inputs.csv',delimiter=',')
  + line 66 - decemberChartInputs.to_csv("december-chart-inputs.csv", index=False)

**THEY MAY NEED READJUSTMENT DEPENDING ON YOUR FILE STRUCTURE**



2. To train the model run 'python modelCode.py' ensure the training data is accessible through 'data/train-test.csv'
3.  running creates 'posted_rate_model.cbm' - the model itself and 'feature_names.json' - containing names of all the features used during training
4. run  'python modelPredictor.py' to run the model and this will create 'validation_predictions.csv' obtained from populating 'validation-predictions-template.csv'
5. run 'python modelDecemberChartInputsPredictor.py' to populate 'december_chart_inputs.csv' 
