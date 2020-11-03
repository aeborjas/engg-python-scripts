import pandas as pd
import os
import glob
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('path', type=Path, default=Path(__file__).absolute().parent, help ='Path to the excel files.')
parser.add_argument('files', type=str, nargs='+')
parser.add_argument('--detail','-d',type=int,help='0 for summary, 1 for detailed', dest='detail')

args = parser.parse_args()

os.chdir(args.path)

if args.detail == 0:
    hdr = 2
elif args.detail == 1:
    hdr = 3
else:
    hdr = 2

df = pd.DataFrame()
for x in args.files:
    print(f'Starting with {x}...')
    if (args.path / x).is_file():
        print(f'{args.path / x} exists.')
        df_temp = pd.ExcelFile(x)
        print(df_temp.sheet_names)
        sheet_name = str(input('What should be the sheet to append?\n>>> '))
        df_temp = pd.read_excel(x,header=hdr, sheet_name=sheet_name)
        df = df.append(df_temp,ignore_index=True)

df.to_excel('appended_output.xlsx')
