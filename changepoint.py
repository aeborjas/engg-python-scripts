import requests, getpass
from bs4 import BeautifulSoup

url = 'https://dynamicrisk.changepointasp.com/?qlink=VTS'

psswd = str(getpass.getpass('Password:'))

payload = {'Email':'armando_borjas@dynamicrisk.net',
           'Password':psswd}

r = requests.post(url, data=payload)
soup = BeautifulSoup(r.content, 'lxml')

print(r)
print(soup)
