import requests
import json

try:
    response = requests.post('http://localhost:5000/run-parameter-analysis', 
                           json={'parameter_type': 'A1_historical_trends'})
    print('Status:', response.status_code)
    print('Response:', json.dumps(response.json(), indent=2))
except Exception as e:
    print('Error:', e) 