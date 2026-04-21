import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

def train_and_save_model():
    # Read Dataset
    df = pd.read_csv("stud.csv")

    # Create Average Score
    df['Avg_Score'] = (
        df['math_score'] +
        df['reading_score'] +
        df['writing_score']
    ) / 3

    # Create Target
    def grade(score):
        if score >= 80:
            return 'High'
        elif score >= 50:
            return 'Medium'
        else:
            return 'Low'

    df['target'] = df['Avg_Score'].apply(grade)

    # Fill Missing Values
    df['race_ethnicity'] = df['race_ethnicity'].fillna(
        df['race_ethnicity'].mode()[0]
    )

    # Features (background features only, as per original code logic)
    # The scores are dropped because they are used to define the target.
    # Predicting target from scores would be trivial (rule-based).
    # The ML model predicts target based on background factors.
    background_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    x = df[background_features]
    y = df['target']

    # One Hot Encoding
    x_encoded = pd.get_dummies(x, drop_first=True)
    training_columns = x_encoded.columns.tolist()

    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded, y_encoded, test_size=0.25, random_state=42
    )

    # Train Model
    model = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(x_train, y_train)

    # Accuracy
    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print("Model Accuracy :-", accuracy)

    # Save components
    joblib.dump(model, 'student_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(training_columns, 'training_columns.pkl')
    
    return model, le, training_columns

def predict_grade(data_dict):
    """
    data_dict: dict with keys gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course
    """
    # Load model and components
    try:
        model = joblib.load('student_model.pkl')
        le = joblib.load('label_encoder.pkl')
        training_columns = joblib.load('training_columns.pkl')
    except:
        model, le, training_columns = train_and_save_model()

    # Create a DataFrame for the input
    input_df = pd.DataFrame([data_dict])
    
    # One-hot encoding for the input
    input_encoded = pd.get_dummies(input_df)
    
    # Align columns with training data
    final_input = pd.DataFrame(columns=training_columns)
    for col in training_columns:
        if col in input_encoded.columns:
            final_input[col] = input_encoded[col]
        else:
            final_input[col] = 0
    
    # Predict
    prediction_encoded = model.predict(final_input)
    prediction = le.inverse_transform(prediction_encoded)
    
    return prediction[0]

def get_categories():
    df = pd.read_csv("stud.csv")
    categories = {
        'gender': df['gender'].unique().tolist(),
        'race_ethnicity': df['race_ethnicity'].unique().tolist(),
        'parental_level_of_education': df['parental_level_of_education'].unique().tolist(),
        'lunch': df['lunch'].unique().tolist(),
        'test_preparation_course': df['test_preparation_course'].unique().tolist()
    }
    return categories

if __name__ == "__main__":
    train_and_save_model()
    print("Model trained and saved.")