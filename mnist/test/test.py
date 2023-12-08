import requests

resp = requests.post("https://mnist-m3n2hltjrq-de.a.run.app/", files={'file': open('asdf.png', 'rb')})

print(resp.json())