import requests, os, re
from pathlib import Path

os.chdir(r"C:\Users\armando_borjas\Documents\Python")

with open('ipc2020_links.txt','r') as f:
    l = f.read()

l = l.split('\n')

headers = {
        "User-Agent": "PostmanRuntime/7.20.1",
        "Accept": "*/*",
        "Cache-Control": "no-cache",
        "Postman-Token": "8eb5df70-4da6-4ba1-a9dd-e68880316cd9,30ac79fa-969b-4a24-8035-26ad1a2650e1",
        "Host": "medianet.edmond-de-rothschild.fr",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "cache-control": "no-cache",
        }

pat = re.compile('(.*)(IPC.*\.pdf)')

directory = 'downloadedIPC2020'

##Path(r'downloadedIPC2020').mkdir(parents=True, exist_ok=True)
if not os.path.exists(directory):
    os.makedirs(directory)
    print('Directory created.')

response = requests.get(l[4],headers=headers)

with open('./downloadedIPC2020/' + pat.match(l[4]).group()[-1], 'wb') as f:
    f.write(response.content)
    print('Downloaded.')


