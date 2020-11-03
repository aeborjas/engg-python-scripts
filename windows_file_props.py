import win32api
import pprint
import os, datetime
import xml.dom.minidom
import pandas as pd, numpy as np, pyodbc

#==============================================================================
def getFileProperties(fname):
#==============================================================================
    """
    Read all properties of the given file return them as a dictionary.
    """
    propNames = ('Comments', 'InternalName', 'ProductName',
        'CompanyName', 'LegalCopyright', 'ProductVersion',
        'FileDescription', 'LegalTrademarks', 'PrivateBuild',
        'FileVersion', 'OriginalFilename', 'SpecialBuild')

    props = {'FixedFileInfo': None, 'StringFileInfo': None, 'FileVersion': None}

    try:
        # backslash as parm returns dictionary of numeric info corresponding to VS_FIXEDFILEINFO struc
        fixedInfo = win32api.GetFileVersionInfo(fname, '\\')
        props['FixedFileInfo'] = fixedInfo
        props['FileVersion'] = "%d.%d.%d.%d" % (fixedInfo['FileVersionMS'] / 65536,
                fixedInfo['FileVersionMS'] % 65536, fixedInfo['FileVersionLS'] / 65536,
                fixedInfo['FileVersionLS'] % 65536)

        # \VarFileInfo\Translation returns list of available (language, codepage)
        # pairs that can be used to retreive string info. We are using only the first pair.
        lang, codepage = win32api.GetFileVersionInfo(fname, '\\VarFileInfo\\Translation')[0]

        # any other must be of the form \StringfileInfo\%04X%04X\parm_name, middle
        # two are language/codepage pair returned from above

        strInfo = {}
        for propName in propNames:
            strInfoPath = u'\\StringFileInfo\\%04X%04X\\%s' % (lang, codepage, propName)
            ## print str_info
            strInfo[propName] = win32api.GetFileVersionInfo(fname, strInfoPath)

        props['StringFileInfo'] = strInfo
    except:
        pass

    return props

def read_config_file(fname):
    doc = xml.dom.minidom.parse(fname)
    dbs = doc.getElementsByTagName("configuration")[0].getElementsByTagName("appSettings")[0].getElementsByTagName('add')
    vers = dict()
    for x in dbs:
        vers.update({x.attributes['key'].value: x.attributes['value'].value})
    return vers

def sqlserver_sql(q, server = 'SQL2017', db = 'plains_irasv6_stage'):
    driver = '{SQL Server Native Client 11.0}'
    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")

    # performing query to database
    df = pd.read_sql_query(q,conn)

    conn.close()
    return df

q = """select * from VersionInfo"""
print(sqlserver_sql(q)['DBVersion'].T)

risk = r"C:\Program Files\Dynamic Risk\RiskAnalyst6\RiskAnalyst.exe"
ilia = r"C:\Program Files\Dynamic Risk\ILIAnalyst6_PPL\ILIAnalyst.exe"
ilia_config = r"C:\Program Files\Dynamic Risk\ILIAnalyst6_PPL\ILIAnalyst.exe.config"

print(getFileProperties(ilia)['FileVersion'], datetime.date.fromtimestamp(os.path.getmtime(ilia)))
pprint.pprint(read_config_file(ilia_config))