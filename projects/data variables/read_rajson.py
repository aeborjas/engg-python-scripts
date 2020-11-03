import json, pandas as pd, os, pprint, argparse
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file', help='risk project file for which to gather information')
args = parser.parse_args()

pd.set_option('display.max_columns',500)

if not args.file:
    os.chdir(r'N:\Python\data variables')
    dpath = r"20191204 DC EEC-US.raprj"
else:
    dpath = args.file

#Extract *.var file from *.raprj
with ZipFile(dpath, 'r') as raprj:
    for x in raprj.namelist():
        if ".rajson" in x:
            name = x[:20]
            with raprj.open(x, 'r') as rajson_doc:
                data = json.load(rajson_doc)


# pprint.pprint(data['ExpectedDataSource'].split('@'))
# temp = dict(num=[],field=[],data=[])
for x, (y,z) in enumerate(data.items()):
    if x <=16:
        print(f"[{x}] {y}:\t{z}")
        # temp['num'].append(x)
        # temp['field'].append(y)
        # temp['data'].append(z)
    else:
        break
# temp = pd.DataFrame(temp)
# temp.set_index('num',drop=True,inplace=True)
# print(temp)