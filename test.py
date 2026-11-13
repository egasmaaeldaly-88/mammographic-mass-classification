import requests

url = 'http://localhost:9696/predict'

cases = {
    'full case': {'age': 60, 'shape': 4, 'margin': 5, 'density': 3},
    'missing values': {'age': 60, 'shape': 4},
}

for name, mass in cases.items():
    response = requests.post(url, json=mass)
    print(name, response.json())