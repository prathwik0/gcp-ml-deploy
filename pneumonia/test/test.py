import requests

resp = requests.post("https://pneumonia-fmaprfvioa-de.a.run.app", files={'file': open('normal1.jpeg', 'rb')})

print(resp.json())