import os, sys, pandas as pd, pyodbc, numpy as np, glob, argparse
from os.path import abspath, dirname, join
from xml.etree import ElementTree as ET
from tkinter import filedialog
import tkinter as tk

parser = argparse.ArgumentParser(description ="enter database, config database, and path to IRAS app to determine if they are compatible.")
parser.add_argument('-st','--stage', help='Stage database')
parser.add_argument('-c','--config', help='Corresponding CONFIG database')
parser.add_argument('-se','--server', help='SQL server')

# args = sys.argv
args = parser.parse_args()

def sql_query(q, db, server):

    driver = '{SQL Server Native Client 11.0}'
    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")
    df = pd.read_sql_query(q,conn)
    conn.close()

    return df

def browseFiles(root):
    filename = filedialog.askopenfilename(parent=root,
                                          initialdir = r"C:\Program Files\Dynamic Risk",
                                          title="Select a File",
                                          filetypes = [("IRAS Configuration File","*.exe.config*")])          
##    label_file_explorer.configure(text="File Opened: "+filename)
    return filename

q = """set nocount on;
        select * from versioninfo"""

stage_ver = sql_query(q, args.stage, args.server)
config_ver = sql_query(q, args.config, args.server)

root = tk.Tk()
root.withdraw()
path = browseFiles(root)
##root.title("File Explorer")
##root.geometry("500x500")
##root.config(background="white")
##label_file_explorer = tk.Label(root,  
##                            text = "File Explorer using Tkinter", 
##                            width = 100, height = 4,  
##                            fg = "blue") 
##button_explore = tk.Button(root,  
##                        text = "Browse Files", 
##                        command = browseFiles)
##button_exit = tk.Button(root,  
##                     text = "Exit", 
##                     command = exit)
### Grid method is chosen for placing 
### the widgets at respective positions  
### in a table like structure by 
### specifying rows and columns 
##label_file_explorer.grid(column = 1, row = 1) 
##button_explore.grid(column = 1, row = 2) 
##button_exit.grid(column = 1,row = 3) 
##   
### Let the window wait for any events 
##root.mainloop() 

info = dict(field=[],value=[])
for x in glob.glob(path):
    xml = ET.parse(x)
    root = xml.getroot()
    for setting in root.find('appSettings'):
        info['field'].append(setting.get('key'))
        info['value'].append(setting.get('value'))

info = pd.DataFrame(info)
info.value = pd.to_numeric(info.value, errors='coerce').dropna()

print(f"DB requirement:\n App: {info.loc[lambda x: x.field == 'MinRequireDatabaseVersion','value'].values[0]}",f"DB: {stage_ver.DBVersion.values[0]}", f"{info.loc[lambda x: x.field == 'MinRequireDatabaseVersion','value'].values[0] <= stage_ver.DBVersion.values[0]}",sep=' <--> ')

print(f"Config DB requirement:\n App: {info.loc[lambda x: x.field == 'MinRequireConfigDatabaseVersion','value'].values[0]}",f"DB: {config_ver.DBVersion.values[0]}", f"{info.loc[lambda x: x.field == 'MinRequireConfigDatabaseVersion','value'].values[0]<=config_ver.DBVersion.values[0]}", sep=' <--> ')
