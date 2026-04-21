import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    'gender': 'female',
    'race_ethnicity': 'group B',
    'parental_level_of_education': "bachelor's degree",
    'lunch': 'standard',
    'test_preparation_course': 'none',
    'math_score': '72',
    'reading_score': '72',
    'writing_score': '74'
}

response = requests.post(url, data=data)
if response.status_code == 200:
    print("Success!")
    if "Score Card Summary" in response.text:
        print("Result card found in response.")
else:
    print(f"Failed with status code {response.status_code}")
    print(response.text)
