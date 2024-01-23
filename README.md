#Thanks for DeepDTA (https://github.com/hkmztrk/DeepDTA)
# 1 create data:
python create_data.py

This returns kiba_train.csv, kiba_test.csv, davis_train.csv, and davis_test.csv, saved in data/ folder. 

## 2. Train a prediction model
To train a model using training data. The model is chosen if it gains the best MSE for testing data.  
Running 

python training.py 0 0 

## 3. Train a prediction model with validation 

python training_validation.py 0 0

