import pandas as pd, numpy as np, pyodbc
from xml.dom import minidom

def sqlserver_sql(q, server = 'SQL2017', db = 'plains_irasv6_stage'):
    driver = '{SQL Server Native Client 11.0}'
    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")

    # performing query to database
    df = pd.read_sql_query(q,conn)

    conn.close()
    return df

q = """select * from VersionInfo"""
sqlserver_sql(q).T

config_author_nov19 = r"C:\Program Files\Dynamic Risk\Author6_nov2019\Author.exe.config"

config_xml =minidom.parse(config_author_nov19)
