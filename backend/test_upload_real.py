import requests

files = {'file': ('output.csv', open('output.csv', 'rb'), 'text/csv')}
data = {'target_column': 'heart_disease_present'}

try:
    response = requests.post('http://127.0.0.1:8000/api/upload', files=files, data=data)
    print("Status:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except:
        print("Response Text:", response.text)
except Exception as e:
    print("Error:", e)
