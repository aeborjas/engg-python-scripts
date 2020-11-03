import json
import pandas as pd
import datetime
import os
import pyodbc
from zipfile import ZipFile

pd.set_option('display.max_columns',500)

os.chdir(r'N:\Python\data variables')

dpath = r"MD Crack Simulation_Rupture_IPL.algproj"

server = 'sql2017'
config_db = 'ipl_irasv6_config'

def get_config_details(db,server):

    driver = '{SQL Server Native Client 11.0}'

    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")

    q0 = """--**************************************************************************
        --This query gathers additional information for the configured fields in the specified database.
        --**************************************************************************
        SELECT --VariableGroupId,
               VariableGroupName,
               --VariableCategoryParentId,
               VariableCategoryParentName,
               --VariableCategoryId,
               VariableCategoryName,
	           TableName,
	           ColumnName,
               --vari.VariableId,
               VariableName,
               Replace( VariableName, '_', ' ' ) AS VariableNameToUpdate,
               VariableDisplayText,
               /*HasVariableMetaData,
               VariableDataType,
               VariableDecimalCount,
               VariableUnitCode,
               HasDatabaseMetaData,*/
	           DataSourceDataType,
               DataTableType/*,
               DataSourceDecimalCount,
               DataSourceUnitCode*/
        FROM   dbo.VariableFieldXRef vx
               INNER JOIN ( SELECT dst.Id    AS DataSourceTableId,
                                   dst.NAME  AS TableName,
                                   dsf.Id    AS DataSourceFieldId,
                                   FieldName AS ColumnName,
                                   dt.NAME   AS DataSourceDataType,
                                   --dm.Id     AS HasDatabaseMetaData,
                                   CASE
                                     WHEN dst.TableType = 2 THEN 'Point'
                                     WHEN dst.TableType = 1 THEN 'Linear'
                                     WHEN dst.TableType = 3 THEN 'Virtual'
                                     ELSE 'Unknown'
                                   END       AS DataTableType--,
                                   --Code      AS DataSourceUnitCode,
                                   --PRECISION AS DataSourceDecimalCount
                            FROM   dbo.DataSourceTable dst
                                   LEFT JOIN dbo.DataSourceField dsf
                                          ON dst.Id = dsf.TableId
                                   --LEFT JOIN dbo.DataSourceFieldMetaData dm
                                   --       ON dsf.Id = dm.FieldId
                                   --LEFT JOIN dbo.UnitOfMeasure uom
                                   --       ON dm.UnitId = uom.Id
                                   LEFT JOIN dbo.DataType dt
                                          ON dsf.DataTypeId = dt.Id ) db
                       ON vx.DataSourceFieldId = db.DataSourceFieldId
               INNER JOIN ( SELECT vg.Id                AS VariableGroupId,
                                   vg.NAME              AS VariableGroupName,
                                   vcp.Id               AS VariableCategoryParentId,
                                   vcp.NAME             AS VariableCategoryParentName,
                                   vc.Id                AS VariableCategoryId,
                                   vc.NAME              AS VariableCategoryName,
                                   v.Id                 AS VariableId,
                                   v.NAME               AS VariableName,
                                   v.DisplayText        AS VariableDisplayText,
                                   dt.NAME              AS VariableDataType--,
                                   --vm.Id                AS HasVariableMetaData,
                                   --vm.DecimalPlaceCount AS VariableDecimalCount,
                                   --uom.Code             AS VariableUnitCode
                            FROM   dbo.Variable v
                                   LEFT JOIN dbo.DataType dt
                                          ON v.DataTypeId = dt.Id
                                   --LEFT JOIN dbo.VariableMetadata vm
                                   --       ON v.Id = vm.VariableId
                                   --LEFT JOIN dbo.UnitOfMeasure uom
                                   --       ON vm.UnitCodeId = uom.Id
                                   LEFT JOIN dbo.VariableGroup vg
                                          ON v.GroupId = vg.Id
                                   LEFT JOIN dbo.VariableCategory vc
                                          ON v.CategoryId = vc.Id
                                   LEFT JOIN dbo.VariableCategory vcp
                                          ON vc.ParentCategoryId = vcp.Id ) vari
                       ON vx.VariableId = vari.VariableId
        --WHERE  VariableGroupName IN ( 'VirtualTable' )
        ORDER  BY DatasourceTableId--VariableGroupName,VariableCategoryParentName,VariableCategoryName,TableName,ColumnName
        """
    df = pd.read_sql_query(q0,conn)

    conn.close()

    return df


now_str = datetime.datetime.now().strftime('%Y-%m-%d')
name = dpath.split("\\")[-1].split(".")[0][:20]

#Extract *.data.config file from *.algproj
with ZipFile(dpath, 'r') as algproj:
    #file = [x for x in algproj.namelist() if ".data.config" in algproj.namelist()]
    for x in algproj.namelist():
        if ".data.config" in x:
            print(x)
            with algproj.open(x, 'r') as data_conf:
                data = json.load(data_conf)

name_con = []
var_name = []
unit_code = []

vars = [name_con,var_name,unit_code]

for x in range(len(data['SegmentationGroups'][0]['Bucket']['Variables'])):
    locals().update(data['SegmentationGroups'][0]['Bucket']['Variables'][x])
    name_con.append(Name)
    var_name.append(Variable)
    unit_code.append(UnitOfMeasure)
print('JSON processed.')

fieldnames = ['Name','MappedVariableName','UnitCode']
data_dict = {x:v for x,v in zip(fieldnames,vars)}
data_df = pd.DataFrame(data_dict)
config_df = get_config_details(config_db,server)

data_df_w_config_info = pd.merge(data_df,config_df, left_on='MappedVariableName', right_on='VariableName', how='outer')

data_df_w_config_info['MappedVariableName'].fillna('****VARIABLE NOT MAPPED****', inplace=True)

ds = input('Export Variables to Excel? (Y/N)')

if str(ds).lower() == 'y':
    writer = pd.ExcelWriter(name+'-data-vars_'+now_str+'.xlsx', engine='xlsxwriter')

    workbook = writer.book
    sh_name = name+'_Inputs'
    data_df_w_config_info.to_excel(writer, sheet_name=sh_name)

    input_ws = writer.sheets[sh_name]

    format1 = workbook.add_format()
    format1.set_font_size(9)
    format1.set_align('vcenter')
    
    format2 = workbook.add_format()
    format2.set_font_size(9)
    format2.set_text_wrap()
    format2.set_align('vcenter')

    format3 = workbook.add_format()
    format3.set_font_size(9)
    format3.set_align('vcenter')
    format3.set_font_color('red')
    format3.set_bg_color('#FFC7CE')
    
    input_ws.autofilter(0,0,0,len(data_df_w_config_info.columns))
    input_ws.freeze_panes(1,1)
    input_ws.set_column('A:Z',None,format1)
    input_ws.set_column('F:F',64,format2)
    input_ws.conditional_format('C1:C1048576',{'type':'cell',
                                            'criteria':'equal to',
                                       'value':'"****VARIABLE NOT MAPPED****"',
                                       'format':format3})

    writer.save()
else:
    pass
