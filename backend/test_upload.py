import urllib.request
import urllib.parse
from urllib.error import HTTPError
import io

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test.csv"\r\n'
    f'Content-Type: text/csv\r\n\r\n'
    f'age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n63,1,3,145,233,1,0,150,0,2.3,0,0,1,1\n37,1,2,130,250,0,1,187,0,3.5,0,0,2,0\r\n'
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="target_column"\r\n\r\n'
    f'target\r\n'
    f'--{boundary}--\r\n'
)

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/upload',
    data=body.encode('utf-8'),
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    method='POST'
)

try:
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        print("Response:", response.read().decode('utf-8'))
except HTTPError as e:
    print("Error:", e.code)
    print("Response:", e.read().decode('utf-8'))
