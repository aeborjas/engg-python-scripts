import sqlite3
import pandas as pd
import pyodbc
import cx_Oracle
import numpy as np
import os

def sqlite_sql(q, path=r"C:\Users\XXXXXX\Documents\XXX_20200317-system.db"):

    # Read sqlite query results into a pandas DataFrame
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(q, conn)

    conn.close()
    return df

def sqlserver_sql(q, server = 'XXXXXX', db = 'XXXX_XXXXXX_XXXXX'):
    driver = '{SQL Server Native Client 11.0}'
    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")

    # performing query to database
    df = pd.read_sql_query(q,conn)

    conn.close()
    return df

def oracle_sql(q,user=r'XXXX_XXXXXX_XXXXX', password='ug', dsn='XXXXXX'):
    conn = cx_Oracle.connect(user, password, dsn)

    t_data = pd.read_sql_query(q,conn)

    conn.close()
    
    return t_data

def get_corr_between_range(begin_point,seg_length,ilir_id,surface='E',result_type='summary'):
    if surface.lower() == "i":
        orientation = ""
    else:
        orientation = "--"
    
    q2 = """SET NOCOUNT ON;
        DECLARE @begin as float = """+str(begin_point)+""",
        @seg as float = """+str(seg_length)+""",
        @end as float,
        @ILIR_id as int = """+str(ilir_id)+""",
        @surf varchar(1) = '"""+surface+"""'

        set		@end = @begin+@seg

        declare @corr_t table (ld varchar(255))
        insert @corr_t(ld) values ('Corrosion affecting Girth Weld'),
        ('Corrosion with Deformation'),
        ('Groove'),
        ('Corrosion affecting Long. Seam'),
        ('Metal Loss'),
        ('Gouge'),
        ('Metal Loss with Crack')

        select ll.linename [line], 
        features.id [FeatureID], 
        ld4.code [vendor], features.ILIFGirthWeldNum [US_GWD_number], 
        features.ILIFGirthWeldNumDS [DS_GWD_number], 
        ld3.code [tool], 
        format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], 
        ld.code [status], 
        ld2.code [type], 
        features.ILIFComment [vendor_comment], 
        features.ILIFFeatureNumber [feature_number], 
        features.ILIFOdometer [feature_ODO], 
        features.StationNum+mlv.FactorNum [IRAS_chainage_m],
        features.ILIFPeakDepthPct [depth_fraction], 
        features.ILIFLength [length_mm], 
        features.ILIFWidth [width_mm], 
        @begin [start_ILI_segment_m], 
        @end [end_ILI_segment_m]  
        from InlineInspectionFeature features
        
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILIFStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILIFTypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId

        where features.InlineInspectionRangeId = @ILIR_id and
        ld.Code = 'Active' and
        ld2.Code in (select * from @corr_t) and
        (features.ILIFSurfaceInd like 'M' or features.ILIFSurfaceInd like 'U' or features.ILIFSurfaceInd like @surf) and
        """+orientation+"""(features.ILIFOrientation between 120 and 240) and
        (features.StationNum+mlv.FactorNum) between @begin and @end
        order by (features.StationNum+mlv.FactorNum) asc
        """

    if result_type == 'summary':
        return sqlserver_sql(q2).describe().loc[['count','min','max'],['InlineInspectionFeatureId','depth_fraction','length_mm']].reset_index().melt(id_vars='index').set_index(['index','variable']).drop([('min','InlineInspectionFeatureId'),('max','InlineInspectionFeatureId'),('count','depth_fraction'),('count','length_mm')]).squeeze()
    else:
        return sqlserver_sql(q2)

def get_corr_by_id(id):
    
    q2 = f"""SET NOCOUNT ON;
        
        select ll.linename [line], 
        features.id [FeatureID],
        ld4.code [vendor],
        ld3.code [tool], 
        format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], 
        ld.code [status], 
        ld2.code [type],
        features.ILIFSurfaceInd [surface],
        features.StationNum+mlv.FactorNum [chainage_m],
        features.ILIFPeakDepthPct [depth_fraction], 
        features.ILIFLength [length_mm], 
        features.ILIFWidth [width_mm], 

        b.[OD_inch],
		b.[WT_mm],
		b.[grade_MPa],
		b.[toughness_J],
		format(b.[install_date],'yyyy-MM-dd') [install_date],
		b.begin_PS,
		b.end_PS,
		d.MAOP_kPa,
		d.begin_maop,
		d.end_maop

        from InlineInspectionFeature features		       
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILIFStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILIFTypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId
   
		join (select ll.id [llid],
				ps.BeginStationSeriesId,
				ps.PipeOutsideDiameter [OD_inch],
				ps.PipeWallThickness [WT_mm],
				ps.PipeGrade [grade_MPa],
				ps.PipeToughness [toughness_J],
				ps.EffectiveStartDate [install_date],
				ps.BeginStationNum+mlv1.FactorNum [begin_PS],
				ps.EndStationNum+mlv2.FactorNum [end_PS] 
				from PipeSegment ps 
			join StationSeries ss on ss.id = ps.BeginStationSeriesId
			join LineLoop ll on ss.LineLoopId = ll.Id
			join MLVCorrection mlv1 on ps.BeginStationSeriesId = mlv1.StationSeriesId
			join MLVCorrection mlv2 on ps.EndStationSeriesId = mlv2.StationSeriesId
			) b on ((features.StationNum+mlv.FactorNum) between b.begin_PS and b.end_PS) and b.llid = ll.id

		join (select ll.id [llid],
				maop.BeginStationSeriesId,
				maop.MaxAllowablePressure [MAOP_kPa],
				maop.BeginStationNum+mlv1.FactorNum [begin_maop],
				maop.EndStationNum+mlv2.FactorNum [end_maop]
				from maop maop
			join StationSeries ss on ss.id = maop.BeginStationSeriesId
			join LineLoop ll on ss.LineLoopId = ll.Id
			join MLVCorrection mlv1 on maop.BeginStationSeriesId = mlv1.StationSeriesId
			join MLVCorrection mlv2 on maop.EndStationSeriesId = mlv2.StationSeriesId
			) d on ((features.StationNum+mlv.FactorNum) between d.[begin_maop] and d.[end_maop]) and d.llid = ll.id

        where features.Id in ({','.join(id.astype(str)) if hasattr(id,'__len__') else str(id)})
        """
    return sqlserver_sql(q2)

def get_cracks_between_range(begin_point,seg_length,ilir_id,model_name,result_type='summary'):
    
    q2 = """SET NOCOUNT ON;
        DECLARE @begin as float = """+str(begin_point)+""",
        @seg as float = """+str(seg_length)+""",
        @end as float,
        @ILIR_id as int = """+str(ilir_id)+""",
        @simulation varchar(1388) = '"""+model_name+"""'	--SCC | MD | CSCC
        
        set		@end = @begin+@seg

        declare @crackscc_t table (ld varchar(255))
        insert @crackscc_t(ld) values ('Crack-Field')

        declare @crackcscc_t table (ld varchar(255))
        insert @crackcscc_t(ld) values ('Circumferential Crack-Field')

        declare @crackmd_t table (ld varchar(255))
        insert @crackmd_t(ld) values ('Crack-Like'),
        ('Notch-Like'),
        ('Long Seam Weld Anomaly'),
        ('Crack-Like Seam')

        select ll.linename [line], 
        features.id [InlineInspectionCrackAnomalyId], 
        ld4.code [vendor], 
        features.ILICAGirthWeldNum [US_GWD_number], 
        features.ILICAGirthWeldNumDS [DS_GWD_number], 
        ld3.code [tool], 
        format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], 
        ld.code [status], 
        ld2.code [type], 
        features.ILICAComment [vendor_comment], 
        features.ILICAAnomalyNumber [feature_number], 
        features.ILICAOdometer [feature_ODO], 
        features.StationNum+mlv.FactorNum [IRAS_chainage_m],
        features.ILICADepthPct [depth_fraction], 
        features.ILICALength [length_mm], 
        features.ILICAWidth [width_mm], 
        @begin [start_ILI_segment_m], 
        @end [end_ILI_segment_m] 
        from InlineInspectionCrackAnomaly features
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILICAStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILICATypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId

        where features.InlineInspectionRangeId = @ILIR_id and
        ld.Code = 'Active' and
        ((@simulation = 'SCC' and ld2.Code in (select * from @crackscc_t)) or
         (@simulation = 'MD' and ld2.Code in (select * from @crackmd_t)) or
         (@simulation = 'CSCC' and ld2.Code in (select * from @crackcscc_t))) and
        (features.StationNum+mlv.FactorNum) between @begin and @end
        order by (features.StationNum+mlv.FactorNum) asc
        """

    if result_type == 'summary':
        return sqlserver_sql(q2).describe().loc[['count','min','max'],['InlineInspectionCrackAnomalyId','depth_fraction','length_mm']].reset_index().melt(id_vars='index').set_index(['index','variable']).drop([('min','InlineInspectionCrackAnomalyId'),('max','InlineInspectionCrackAnomalyId'),('count','depth_fraction'),('count','length_mm')]).squeeze()
    else:
        return sqlserver_sql(q2)

def get_cracks_by_id(id):
    
    q2 = f"""SET NOCOUNT ON;
        
        select ll.linename [line], 
        features.id [FeatureID], 
        ld4.code [vendor], 
        ld3.code [tool], 
        format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], 
        ld.code [status], 
        ld2.code [type],
        features.ILICASurfaceInd [surface], 
        features.StationNum+mlv.FactorNum [chainage_m],
        features.ILICADepthPct [depth_fraction], 
        features.ILICALength [length_mm], 
        features.ILICAWidth [width_mm], 
        
        b.[OD_inch],
		b.[WT_mm],
		b.[grade_MPa],
		b.[toughness_J],
		format(b.[install_date],'yyyy-MM-dd') [install_date],
		b.begin_PS,
		b.end_PS,
		d.MAOP_kPa,
		d.begin_maop,
		d.end_maop
		
        from InlineInspectionCrackAnomaly features
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILICAStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILICATypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId
   
		join (select ll.id [llid],
				ps.BeginStationSeriesId,
				ps.PipeOutsideDiameter [OD_inch],
				ps.PipeWallThickness [WT_mm],
				ps.PipeGrade [grade_MPa],
				ps.PipeToughness [toughness_J],
				ps.EffectiveStartDate [install_date],
				ps.BeginStationNum+mlv1.FactorNum [begin_PS],
				ps.EndStationNum+mlv2.FactorNum [end_PS] 
				from PipeSegment ps 
			join StationSeries ss on ss.id = ps.BeginStationSeriesId
			join LineLoop ll on ss.LineLoopId = ll.Id
			join MLVCorrection mlv1 on ps.BeginStationSeriesId = mlv1.StationSeriesId
			join MLVCorrection mlv2 on ps.EndStationSeriesId = mlv2.StationSeriesId
			) b on ((features.StationNum+mlv.FactorNum) between b.begin_PS and b.end_PS) and b.llid = ll.id

		join (select ll.id [llid],
				maop.BeginStationSeriesId,
				maop.MaxAllowablePressure [MAOP_kPa],
				maop.BeginStationNum+mlv1.FactorNum [begin_maop],
				maop.EndStationNum+mlv2.FactorNum [end_maop]
				from maop maop
			join StationSeries ss on ss.id = maop.BeginStationSeriesId
			join LineLoop ll on ss.LineLoopId = ll.Id
			join MLVCorrection mlv1 on maop.BeginStationSeriesId = mlv1.StationSeriesId
			join MLVCorrection mlv2 on maop.EndStationSeriesId = mlv2.StationSeriesId
			) d on ((features.StationNum+mlv.FactorNum) between d.[begin_maop] and d.[end_maop]) and d.llid = ll.id

        where features.Id in ({','.join(id.astype(str)) if hasattr(id,'__len__') else str(id)})
        """

    return sqlserver_sql(q2)

if __name__ == '__main__':
    ids = pd.read_csv(path)
    print(get_cracks_by_id(ids.Id))
