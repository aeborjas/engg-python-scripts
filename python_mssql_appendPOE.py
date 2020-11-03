import pyodbc
import pandas as pd
import time
import os
import numpy as np
import glob

pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr',True)
# pd.set_option('display.width', 500)


def get_poe_for_id(ids, model):
    server = 'sql2017'

    db = 'plains_irasv5_stage'
    driver = '{SQL Server Native Client 11.0}'

    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")

    q1 = """set nocount on;
        declare @temp_t table (idnum numeric(10,0))
        
        insert into @temp_t(idnum)
        select * from (values """+ids+""") A(idnum)

        declare @model varchar(255) = '"""+model+"""' --corrosion | NCD | crackSCC | crackMD | crackCSCC | inferredSCC | inferredCSCC
        
        IF @model = 'corrosion'
        select mc.InlineInspectionFeatureId, mc.Algorithm, mc.FailureProbability, mc.LeakProbability, mc.RuptureProbability, mc.AnomalyPeakDepthPct [MC_depth_pct], mc.AnomalyLength [MC_length_mm] from MonteCarlo_Feature mc
        where mc.InlineInspectionFeatureId in (select * from @temp_t)
        and mc.Algorithm like 'Type_E_Simulation_CGA_Range'

        ELSE IF @model = 'NCD'
        select mc.InlineInspectionFeatureId, mc.Algorithm, mc.FailureProbability, mc.LeakProbability, mc.RuptureProbability, mc.AnomalyPeakDepthPct [MC_depth_pct], mc.AnomalyLength [MC_length_mm] from MonteCarlo_NCD mc
        where mc.InlineInspectionFeatureId in (select * from @temp_t)
        and mc.Algorithm like 'Mechanical_Damage_Resident'

        else if @model = 'crackSCC'
        select mc.InlineInspectionFeatureId, mc.Algorithm, mc.FailureProbability, mc.LeakProbability, mc.RuptureProbability, mc.AnomalyPeakDepthPct [MC_depth_pct], mc.AnomalyLength [MC_length_mm] from MonteCarlo_Crack mc
        where mc.InlineInspectionFeatureId in (select * from @temp_t)
        and mc.Algorithm like 'Crack_POF_SCC'

        else if @model = 'crackMD'
        select mc.InlineInspectionFeatureId, mc.Algorithm, mc.FailureProbability, mc.LeakProbability, mc.RuptureProbability, mc.AnomalyPeakDepthPct [MC_depth_pct], mc.AnomalyLength [MC_length_mm] from MonteCarlo_Crack mc
        where mc.InlineInspectionFeatureId in (select * from @temp_t)
        and mc.Algorithm like 'Crack_POF_MD'

        else if @model = 'crackCSCC'
        select mc.InlineInspectionFeatureId, mc.Algorithm, mc.FailureProbability, mc.LeakProbability, mc.RuptureProbability, mc.AnomalyPeakDepthPct [MC_depth_pct], mc.AnomalyLength [MC_length_mm] from MonteCarlo_Crack mc
        where mc.InlineInspectionFeatureId in (select * from @temp_t)
        and mc.Algorithm like 'Crack_POF_CSCC'

        else if @model = 'inferredSCC'
        select mc.InlineInspectionFeatureId, mc.Algorithm, mc.FailureProbability, mc.LeakProbability, mc.RuptureProbability, mc.AnomalyPeakDepthPct [MC_depth_pct], mc.AnomalyLength [MC_length_mm] from MonteCarlo_Feature mc
        where mc.InlineInspectionFeatureId in (select * from @temp_t)
        and mc.Algorithm like 'Inferred_Crack_POF_SCC'

        else if @model = 'inferredCSCC'
        select mc.InlineInspectionFeatureId, mc.Algorithm, mc.FailureProbability, mc.LeakProbability, mc.RuptureProbability, mc.AnomalyPeakDepthPct [MC_depth_pct], mc.AnomalyLength [MC_length_mm] from MonteCarlo_Feature mc
        where mc.InlineInspectionFeatureId in (select * from @temp_t)
        and mc.Algorithm like 'Inferred_Crack_POF_CSCC'"""

    df = pd.read_sql_query(q1, conn)
    conn.close()
    return df


def rename_id_column(df):
    df = df.rename(columns={ df.columns[2] :'InlineInspectionFeatureId'})
    return df


if __name__ == '__main__':
    
    main_path = r'Z:\Plains Midstream\2019_01_Annual Risk Data Support\3_Engineering\Results & QC\3 Risk Drivers\20190208-430p-system'
    threat_path = r'\Environment\MD'
    folder_path = main_path + threat_path

    script_stime = time.time()
    os.chdir(folder_path)

    model = 'crackMD'     #corrosion | NCD | crackSCC | crackMD | crackCSCC | inferredSCC | inferredCSCC

    print(f'\n{len(glob.glob("*"))} items\n','\n'.join(glob.glob('*')))
    input('\nPress key to continue...')

    for x in glob.glob('*.csv'):
        file_stime = time.time()
        edf = pd.DataFrame()
        df = pd.read_csv(x)
        try:
            df['InlineInspectionFeatureId']
        except KeyError:
            df = rename_id_column(df)
        
        print(f'{df.shape[0]} features -- {x}')
        ids = df['InlineInspectionFeatureId'].values
        str_ids = ["("+str(id)+")" for id in list(ids)]
        str_ids = ",".join(str_ids)

        # print(str_ids)    #print string ids for QC purposes

        rdf = get_poe_for_id(str_ids,model)
        edf = edf.append(rdf)

        edf.reset_index(drop=True, inplace=True)
    
        result = pd.merge(df,edf, on='InlineInspectionFeatureId')

        # print(result)     #print result for QC purposes

        result.to_csv(x)
        file_total_time = time.time() - file_stime
        print(f"\tFinished -- {result.shape[0]} features -- {file_total_time} seconds.")

    script_total_time = time.time() - script_stime

    print(f"Finished script. Total time: {script_total_time} seconds.")

    
