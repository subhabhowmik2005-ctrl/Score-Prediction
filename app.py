from flask import Flask, render_template, request, jsonify
import model
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    categories = model.get_categories()
    return render_template('index.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'gender': request.form.get('gender'),
            'race_ethnicity': request.form.get('race_ethnicity'),
            'parental_level_of_education': request.form.get('parental_level_of_education'),
            'lunch': request.form.get('lunch'),
            'test_preparation_course': request.form.get('test_preparation_course')
        }
        
        # Numeric scores
        math_score = float(request.form.get('math_score', 0))
        reading_score = float(request.form.get('reading_score', 0))
        writing_score = float(request.form.get('writing_score', 0))
        
        # Calculate actual stats from inputs
        total_score = math_score + reading_score + writing_score
        avg_score = total_score / 3
        
        # Determine grade based on inputs (the truth)
        if avg_score >= 80:
            actual_grade = 'High'
        elif avg_score >= 50:
            actual_grade = 'Medium'
        else:
            actual_grade = 'Low'

        # Use the ML model to predict based on background features
        predicted_grade = model.predict_grade(data)
        
        result = {
            'math': math_score,
            'reading': reading_score,
            'writing': writing_score,
            'total': round(total_score, 2),
            'average': round(avg_score, 2),
            'actual_grade': actual_grade,
            'predicted_grade': predicted_grade
        }
        
        return render_template('index.html', result=result, categories=model.get_categories(), scroll_to_result=True)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
