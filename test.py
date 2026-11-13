import requests

url = 'http://localhost:9696/predict'

cases = {
    'low-risk pattern (round, circumscribed, age 40)': {'age': 40, 'shape': 1, 'margin': 1, 'density': 3},
    'high-risk pattern (irregular, spiculated, age 70)': {'age': 70, 'shape': 4, 'margin': 5, 'density': 3},
    'high-risk shape, young patient': {'age': 30, 'shape': 4, 'margin': 5, 'density': 3},
    'missing values (age and shape only)': {'age': 60, 'shape': 4},
}

for name, mass in cases.items():
    response = requests.post(url, json=mass)
    print(name, '->', response.json())