import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("C:\Users\30385\Desktop\yue_celadon_plugin\data", "rb")}
response = requests.post(url, files=files)
print(response.json())