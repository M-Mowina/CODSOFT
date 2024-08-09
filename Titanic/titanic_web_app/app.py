from flask import Flask, render_template, request, redirect, session, flash
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib


# Scale the input features
scaler = joblib.load('../titanic_scaler.joblib')

# Load the image classification model
model = tf.keras.models.load_model('../titanic_nn_model.h5')

# Questionnair columns
columns = ['SibSp', 'Pclass', 'Parch', 'Fare', 'Age', 'Sex', 'Embarked_C',
       'Embarked_Q', 'Embarked_S']

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def survival():
  if request.method == 'POST':
    inputs = {
       'SibSp':[request.form['SibSp']],
       'Pclass': [request.form['Pclass']], 
       'Parch': [request.form['Parch']], 
       'Fare': [request.form['Fare']], 
       'Age':[request.form['Age']], 
       'Sex':[request.form['Sex']],
       'Embarked':[request.form['Embarked']]
       }

    x_new = pd.DataFrame(inputs)
    print(x_new)
    print()

    # Process user responses and return results
    prediction = predict(x_new)
    if prediction[0][0] > 0.5:
       result = 'YOU SURVIVED'
    else: result = 'DID NOT SURVIVED'
    
    print('Result: ',result[0][0])
    return render_template('result.html', result=result)
  else:
    return render_template('survival.html')



def preprocess_input(input_data, scaler = scaler, columns = columns):

    # One-hot encode the input data
    input_data = pd.get_dummies(input_data, columns=['Embarked'])
    
    # Ensure input_data has the same columns as the training data
    missing_cols = set(columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[columns]
    
    # Scale the input data
    input_data[['Age', 'Fare']] = scaler.transform(input_data[['Age', 'Fare']])
    #input_data = pd.DataFrame(scaler.transform(input_data), columns=columns, index=input_data.index)

    return input_data

def predict(input_data, model = model, scaler = scaler, columns = columns):

    # Preprocess the input data
    input_data_preprocessed = preprocess_input(input_data, scaler, columns).astype('float32')
    
    # Make predictions
    predictions = model.predict(input_data_preprocessed)
    
    return predictions


if __name__ == '__main__':
    app.run()
