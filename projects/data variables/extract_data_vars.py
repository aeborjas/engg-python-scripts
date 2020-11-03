import json
import pandas as pd
import datetime
import os
import pyodbc
from zipfile import ZipFile

pd.set_option('display.max_columns',500)

os.chdir(r'C:\Users\armando_borjas\Documents\Python\projects\data variables')

dpath = r"PMC_20200406T1600-selected.raprj"

server = 'sql2017'
config_db = 'PLAINS_IRASv6_CONFIG_PROD'

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

def conv_list_to_str(l):
    o = ''
    for x in l:
        o = o+str(x)+'\n'
    return o.encode()

def parse_data_type_in(number):
    if number == 1:
        return 'Numeric'
    elif number == 2:
        return 'Datetime'
    elif number == 3:
        return 'String'
    else:
        pass

def parse_conf_var_in(string):
    if not(string):
        return 'Data Configuration Variable.'
    else:
        pass

#The following function is a starting point to gather domain catalogs
def get_domaincatalog(string):
    """select * from
        (
        select ld.id [ListDomainID] ,ld.code, ld.LastModDateTime, xr.TableName, xr.ColumnName from ListDomain ld
        join DomainCatalog dc on ld.DomainCatalogId = dc.Id
        join DomainCatalogColXRef xr on dc.id = xr.DomainCatalogId
        where dc.id in (
			select xr.DomainCatalogId from DomainCatalogColXRef xr
			where xr.ColumnName in
			('ClassAreaRatingDomainId',
			'ProductDomainId',
			'HabitatWildlifeSensitivityCat',
			'EA_SCCSeverity',
			'PipeSeamDomainId',
			'EA_LFERWSeverity',
			'LandUseDomainId',
			'MDPublicAwareness',
			'MDSignage',
			'MDBuriedMarkers',
			'MDThirdPartyNotification',
			'MDPatrolFrequency',
			'MDOperatorResponse',
			'MDPipeFinding',
			'MDPipeMarking',
			'DepthofCoverRange',
			'EnvironmentMedium',
			'CoatingLineDomainId',
			'PipeMaterialDomainId',
			'ValveOperatorTypeDomainId',
			'ProductTypeDomainId',
			'ILIRStatusDomainId',
			'ILIFStatusDomainId',
			'ILIFTypeDomainId',
			'ILIRToolDomainId',
			'ILICAStatusDomainId',
			'ILICATypeDomainId',
			'ThreatTypeDomainId') and xr.TableName not in ('IncidentHistory','Stg_ILICrackAnomaly'))
			) b
		where b.ColumnName = 'threattypedomainid'
		order by b.TableName asc, b.ColumnName asc"""
        

now_str = datetime.datetime.now().strftime('%Y-%m-%d')
#name = dpath.split("\\")[-1].split(".")[0][:20]

#Extract *.var file from *.raprj
with ZipFile(dpath, 'r') as raprj:
    for x in raprj.namelist():
        if ".var" in x:
            name = x[:20]
            with raprj.open(x, 'r') as var_doc:
                data = json.load(var_doc)

#print('loading data...')
#with open(dpath,'r') as read_file:
#    data = json.load(read_file)
#print('json loaded.')

var_type_in = []
alg_in_name_in = []
var_name_in = []
unit_code_in = []
consumed_by_in = []
data_type_in = []
conf_var_in = []

vars = [alg_in_name_in,var_name_in,var_type_in,unit_code_in,consumed_by_in,data_type_in,conf_var_in]

print('processing lists...')
#for x in range(len(data['InputVar'])):
#    alg_in_name_in.append(data['InputVar'][x]['AlgorithmInputName'])
#    var_name_in.append(data['InputVar'][x]['VariableName'])
#print('lists processed.')

for x in range(len(data['InputVar'])):
    locals().update(data['InputVar'][x])
    var_type_in.append('Input')
    alg_in_name_in.append(AlgorithmInputName)
    var_name_in.append(VariableName)
    unit_code_in.append(UnitCode)
    consumed_by_in.append(conv_list_to_str(ConsumedBy).decode('utf-8'))
    data_type_in.append(parse_data_type_in(DataType))
    conf_var_in.append(parse_conf_var_in(IsCustomMappingSupported))
print('json processed.')


fieldnames = ['AlgorithmInputName','MappedVariableName','Variable Type','UnitCode','UsedWithinModule','DataType','DataConfigVariable']

data_dict_in = {x:v for x,v in zip(fieldnames, vars)}

data_df_in = pd.DataFrame(data_dict_in)
config_df = get_config_details(config_db,server)

data_df_w_config_info = pd.merge(data_df_in,config_df, left_on='MappedVariableName', right_on='VariableName', how='outer')

data_df_w_config_info['MappedVariableName'].fillna('****VARIABLE NOT MAPPED****', inplace=True)

var_type_o = []
output_name_o = []
var_name_o = []
unit_code_o = []
par_var_name_o = []
weighted_o = []
data_type_o = []
precision_o = []
is_shrd_o = []
consumed_by_o = []
assigned_by_o = []
sec_o = []

vars_o = [output_name_o,var_name_o,var_type_o,unit_code_o,consumed_by_o,data_type_o,is_shrd_o,sec_o]

print('processing lists...')

for x in range(len(data['OutputVar'])):
    locals().update(data['OutputVar'][x])
    var_type_o.append('Output')
    output_name_o.append(OutputName)
    var_name_o.append(VariableName)
    unit_code_o.append(UnitCode)
    par_var_name_o.append(ParentVariableName)
    weighted_o.append(Weighting)
    data_type_o.append(parse_data_type_in(DataType))
    is_shrd_o.append(IsGeneratedBySharedModule)
    consumed_by_o.append(conv_list_to_str(ConsumedByModules).decode('utf-8'))
    assigned_by_o.append(AssignedByModule)
    if OutputName != AssignedByModule:
        sec_o.append("SECONDARY OUTPUT")
    else:
        sec_o.append("")
print('json processed.')

fieldnames_o = ["OutputName","VariableName","Variable Type","UnitCode","UsedWithinModule","DataType","IsGeneratedBySharedModule","IsSecondaryOutput"]

data_dict_o = {x:v for x,v in zip(fieldnames_o, vars_o)}

data_df_o = pd.DataFrame(data_dict_o)

ds = input('Export Variables to Excel? (Y/N)')

if str(ds).lower() == 'y':
    writer = pd.ExcelWriter(name+'-data-vars_'+now_str+'.xlsx', engine='xlsxwriter')

    workbook = writer.book
    
    data_df_w_config_info.to_excel(writer, sheet_name='Inputs')
    data_df_o.to_excel(writer, sheet_name='Outputs')

    input_ws = writer.sheets['Inputs']
    output_ws = writer.sheets['Outputs']

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

    output_ws.autofilter(0,0,0,len(data_df_o.columns))
    output_ws.freeze_panes(1,1)
    output_ws.set_column('A:Z',None,format1)
    output_ws.set_column('F:F',64,format2)

    writer.save()
else:
    pass
