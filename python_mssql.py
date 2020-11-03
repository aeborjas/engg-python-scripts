import pyodbc
import pandas as pd
import time
import os
from os.path import abspath, dirname, join
import numpy as np

###CSV_PATH points to a csv with 3 columns 'Begin Measure (m)', 'End Measure(m)', 'Length (m)'
###Each row represents a dynamic segment that is high likelihood
###Two methods are used here:
######Method 1 Use accesses the database, and queries out the features between the Being Measure (m) and Begin Measure (m)+Length (m)
######Method 2 Calculates the fixed length ILI segments (at maximum length of 30m), normalizes and collapses the dynamic segments from CSV_PATH to the ILI segments, accesses the database, and queries our the features in those ILI segments
##database connection strings are setup within each function, so if the server/database is to be changed, it has to be harcoded

pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr',True)
##pd.set_option('display.width', 500)

def get_ILI_segs(line_name, max_length=30):
    server = 'sql2017'
    
    db = 'plains_irasv6_stage'
    driver = '{SQL Server Native Client 11.0}'

    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")

    q0 = """set nocount on;
        DECLARE @line varchar(255) = '%"""+line_name+"""%'
        select sum(abs(ss.EndStationNum-ss.BeginStationNum)) [length_m] from StationSeries ss
        join LineLoop ll on ss.LineLoopId = ll.Id
        where ll.LineName like @line and ss.EffectiveEndDate is null"""
    df_len = pd.read_sql_query(q0,conn)

    conn.close()
    
    line_len = df_len['length_m'].values[0]
    ili_seg_cnt = np.ceil(line_len/max_length)
    ili_seg_len = line_len/ili_seg_cnt
    ili_segs = np.arange(0,line_len,ili_seg_len)
    return ili_segs, ili_seg_len,line_len

def normalize_to_ili_seg(begseg, ilisegs):
    out = []
    for x in begseg:
        temp = np.extract(x>=np.round(ilisegs,decimals=3),ilisegs)[-1]
        out.append(temp)

    out = np.unique(np.round(np.array(out), decimals=3))
    return out

def get_ILIR(line_name):
    server = 'sql2017'
    
    db = 'plains_irasv6_stage'
    driver = '{SQL Server Native Client 11.0}'

    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")

    q1 = """set nocount on;
        DECLARE @line varchar(255) = '%"""+line_name+"""%'
        select ll.linename [line], ilir.id [ILIR_ID], ilir.EffectiveStartDate, ilir.EffectiveEndDate, ld2.code [status], ld.code [tool], ilir.ILIRStartDate, ilir.ILIRDescr, ilir.BeginStationSeriesId, ilir.BeginStationNum, ilir.EndStationSeriesId, ilir.EndStationNum, ilir.ILIRStartOdometer, ilir.ILIREndOdometer from InlineInspectionRange ilir
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on ilir.ILIRToolDomainId = ld.Id
        join ListDomain ld2 on ilir.ILIRStatusDomainId = ld2.Id
        where ll.LineName like @line
        order by ilir.ILIRStartDate desc"""

    df_ili = pd.read_sql_query(q1,conn)
    print(df_ili)
    ilir_id = input('which ILI to chose?: ')

    conn.close()

    return int(ilir_id)

def get_features_between_range(line_name,begin_point,seg_length,ilir_id,model_name,surface='E'):
    server = 'sql2017'
    
    db = 'plains_irasv6_stage'
    driver = '{SQL Server Native Client 11.0}'

    conn = pyodbc.connect("Driver="+driver+";Server="+server+";Database="+db+";Trusted_Connection=yes;")

    if surface.lower() == "i":
        orientation = ""
    else:
        orientation = "--"
    
    q2 = """SET NOCOUNT ON;
        DECLARE @line varchar(255) = '%"""+line_name+"""%',
        @begin as float = """+str(begin_point)+""",
        @seg as float = """+str(seg_length)+""",
        @end as float,
        @ILIR_id as int = """+str(ilir_id)+""",
        @simulation varchar(1388) = '"""+model_name+"""',	--corrosion | NCD | crackSCC | crackMD | crackCSCC
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

        declare @ncd_t table (ld varchar(255))
        insert @ncd_t(ld) values ('Deformation'),
        ('Sharp Deformation'),
        ('Deformation at Stress Riser'),
        ('Deformation at Girth Weld'),
        ('Deformation on Long Seam'),
        ('Deformation with Metal Loss'),
        ('Gouge'),
        ('Deformation with Arc Burn'),
        ('Deformation with Crack'),
        ('Deformation with Gouge'),
        ('Deformation with Groove')

        declare @crackscc_t table (ld varchar(255))
        insert @crackscc_t(ld) values ('Crack-Field')

        declare @crackcscc_t table (ld varchar(255))
        insert @crackcscc_t(ld) values ('Circumferential Crack-Field')

        declare @crackmd_t table (ld varchar(255))
        insert @crackmd_t(ld) values ('Crack-Like'),
        ('Notch-Like'),
        ('Long Seam Weld Anomaly'),
        ('Crack-Like Seam')

        IF @simulation = 'corrosion'
        select ll.linename [line], features.id [InlineInspectionFeatureId], ld4.code [vendor], features.ILIFGirthWeldNum [US_GWD_number], features.ILIFGirthWeldNumDS [DS_GWD_number], ld3.code [tool], format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], ld.code [status], ld2.code [type], features.ILIFComment [vendor_comment], features.ILIFFeatureNumber [feature_number], features.ILIFOdometer [feature_ODO], features.StationNum+mlv.FactorNum [IRAS_chainage_m],features.ILIFPeakDepthPct [depth_fraction], features.ILIFLength [length_mm], features.ILIFWidth [width_mm], b.[OD_inch], b.[WT_mm], b.[grade_MPa], b.[toughness_J], format(b.[install_date],'yyyy-MM-dd') [install_date], d.MAOP_kPa, b.begin_PS, b.end_PS, d.begin_maop, d.end_maop, @begin [start_ILI_segment_m], @end [end_ILI_segment_m], mc.Algorithm [MCAlgorithm], mc.FailureProbability [MCPOE], mc.LeakProbability [MCLPOE], mc.RuptureProbability [MCRPOE]  from InlineInspectionFeature features
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILIFStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILIFTypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId

        join ( select ll.id [llid], ps.BeginStationSeriesId, ps.PipeOutsideDiameter [OD_inch], ps.PipeWallThickness [WT_mm], ps.PipeGrade [grade_MPa], ps.PipeToughness [toughness_J], ps.EffectiveStartDate [install_date], ps.BeginStationNum+mlv1.FactorNum [begin_PS], ps.EndStationNum+mlv2.FactorNum [end_PS]  from PipeSegment ps 
        join StationSeries ss on ss.id = ps.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on ps.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on ps.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, ps.BeginStationSeriesId asc*/  ) b on ((features.StationNum+mlv.FactorNum) between b.begin_PS and b.end_PS) and b.llid = ll.id

        join ( select ll.id [llid], maop.BeginStationSeriesId, maop.MaxAllowablePressure [MAOP_kPa], maop.BeginStationNum+mlv1.FactorNum [begin_maop], maop.EndStationNum+mlv2.FactorNum [end_maop] from maop maop
        join StationSeries ss on ss.id = maop.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on maop.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on maop.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, maop.BeginStationSeriesId asc*/  ) d on ((features.StationNum+mlv.FactorNum) between d.[begin_maop] and d.[end_maop]) and d.llid = ll.id 

        join MonteCarlo_Feature mc on features.id = mc.InlineInspectionFeatureId and mc.Algorithm like '%Type_E_Simulation_CGA_Range%'

        where ll.LineName like '%'+@line+'%' and 
        features.InlineInspectionRangeId = @ILIR_id and
        ld.Code = 'Active' and
        ld2.Code in (select * from @corr_t) and
        (features.ILIFSurfaceInd like 'M' or features.ILIFSurfaceInd like 'U' or features.ILIFSurfaceInd like @surf) and
        """+orientation+"""(features.ILIFOrientation between 120 and 240) and
        (features.StationNum+mlv.FactorNum) between @begin and @end
        order by (features.StationNum+mlv.FactorNum) asc
        ELSE IF @simulation = 'NCD'
        select ll.linename [line], features.id [InlineInspectionNCDId], ld4.code [vendor], features.ILINCDGirthWeldNum [US_GWD_number], features.ILINCDGirthWeldNumDS [DS_GWD_number], ld3.code [tool], format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], ld.code [status], ld2.code [type], features.ILINCDComment [vendor_comment], features.ILINCDFeatureNumber [feature_number], features.ILINCDOdometer [feature_ODO], features.StationNum+mlv.FactorNum [IRAS_chainage_m],features.ILINCDDepthPctNPS [depth_fraction], features.ILINCDLength [length_mm], features.ILINCDWidth [width_mm], b.[OD_inch], b.[WT_mm], b.[grade_MPa], b.[toughness_J], format(b.[install_date],'yyyy-MM-dd') [install_date], d.MAOP_kPa, b.begin_PS, b.end_PS, d.begin_maop, d.end_maop, @begin [start_ILI_segment_m], @end [end_ILI_segment_m], mc.Algorithm [MCAlgorithm], mc.FailureProbability [MCPOE], mc.LeakProbability [MCLPOE], mc.RuptureProbability [MCRPOE] from InlineInspectionNonCrsnDfct features
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILINCDStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILINCDTypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId

        join ( select ll.id [llid], ps.BeginStationSeriesId, ps.PipeOutsideDiameter [OD_inch], ps.PipeWallThickness [WT_mm], ps.PipeGrade [grade_MPa], ps.PipeToughness [toughness_J], ps.EffectiveStartDate [install_date], ps.BeginStationNum+mlv1.FactorNum [begin_PS], ps.EndStationNum+mlv2.FactorNum [end_PS]  from PipeSegment ps 
        join StationSeries ss on ss.id = ps.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on ps.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on ps.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, ps.BeginStationSeriesId asc*/ ) b on ((features.StationNum+mlv.FactorNum) between b.begin_PS and b.end_PS) and b.llid = ll.id

        join ( select ll.id [llid], maop.BeginStationSeriesId, maop.MaxAllowablePressure [MAOP_kPa], maop.BeginStationNum+mlv1.FactorNum [begin_maop], maop.EndStationNum+mlv2.FactorNum [end_maop] from maop maop
        join StationSeries ss on ss.id = maop.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on maop.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on maop.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, maop.BeginStationSeriesId asc*/ ) d on ((features.StationNum+mlv.FactorNum) between d.[begin_maop] and d.[end_maop]) and d.llid = ll.id

        join MonteCarlo_NCD mc on features.id = mc.InlineInspectionFeatureId and mc.Algorithm like '%Mechanical_Damage_Resident%'

        where ll.LineName like '%'+@line+'%' and 
        features.InlineInspectionRangeId = @ILIR_id and
        ld.Code = 'Active' and
        ld2.Code in (select * from @ncd_t) and
        --ld2.Code in ('Sharp Deformation','Deformation at Girth Weld','Deformation on Long Seam','Deformation with Metal Loss','Gouge','Ovality','Deformation','Deformation at Stress Riser') and
        (features.StationNum+mlv.FactorNum) between @begin and @end
        order by (features.StationNum+mlv.FactorNum) asc
        ELSE IF @simulation = 'crackSCC'
        select ll.linename [line], features.id [InlineInspectionCAId], ld4.code [vendor], features.ILICAGirthWeldNum [US_GWD_number], features.ILICAGirthWeldNumDS [DS_GWD_number], ld3.code [tool], format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], ld.code [status], ld2.code [type], features.ILICAComment [vendor_comment], features.ILICAAnomalyNumber [feature_number], features.ILICAOdometer [feature_ODO], features.StationNum+mlv.FactorNum [IRAS_chainage_m],features.ILICADepthPct [depth_fraction], features.ILICALength [length_mm], features.ILICAWidth [width_mm], b.[OD_inch], b.[WT_mm], b.[grade_MPa], b.[toughness_J], format(b.[install_date],'yyyy-MM-dd') [install_date], d.MAOP_kPa, b.begin_PS, b.end_PS, d.begin_maop, d.end_maop, @begin [start_ILI_segment_m], @end [end_ILI_segment_m], mc.Algorithm [MCAlgorithm], mc.FailureProbability [MCPOE], mc.LeakProbability [MCLPOE], mc.RuptureProbability [MCRPOE] from InlineInspectionCrackAnomaly features
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILICAStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILICATypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId

        join ( select ll.id [llid], ps.BeginStationSeriesId, ps.PipeOutsideDiameter [OD_inch], ps.PipeWallThickness [WT_mm], ps.PipeGrade [grade_MPa], ps.PipeToughness [toughness_J], ps.EffectiveStartDate [install_date], ps.BeginStationNum+mlv1.FactorNum [begin_PS], ps.EndStationNum+mlv2.FactorNum [end_PS]  from PipeSegment ps 
        join StationSeries ss on ss.id = ps.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on ps.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on ps.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, ps.BeginStationSeriesId asc*/  ) b on ((features.StationNum+mlv.FactorNum) between b.begin_PS and b.end_PS) and b.llid = ll.id

        join ( select ll.id [llid], maop.BeginStationSeriesId, maop.MaxAllowablePressure [MAOP_kPa], maop.BeginStationNum+mlv1.FactorNum [begin_maop], maop.EndStationNum+mlv2.FactorNum [end_maop] from maop maop
        join StationSeries ss on ss.id = maop.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on maop.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on maop.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, maop.BeginStationSeriesId asc*/ ) d on ((features.StationNum+mlv.FactorNum) between d.[begin_maop] and d.[end_maop]) and d.llid = ll.id

        join MonteCarlo_Crack mc on features.id = mc.InlineInspectionFeatureId and mc.Algorithm like '%Crack_POF_SCC%'

        where ll.LineName like '%'+@line+'%' and 
        features.InlineInspectionRangeId = @ILIR_id and
        ld.Code = 'Active' and
        ld2.Code in (select * from @crackscc_t) and
        (features.StationNum+mlv.FactorNum) between @begin and @end
        order by (features.StationNum+mlv.FactorNum) asc
        ELSE IF @simulation = 'crackCSCC'
        select ll.linename [line], features.id [InlineInspectionCAId], ld4.code [vendor], features.ILICAGirthWeldNum [US_GWD_number], features.ILICAGirthWeldNumDS [DS_GWD_number], ld3.code [tool], format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], ld.code [status], ld2.code [type], features.ILICAComment [vendor_comment], features.ILICAAnomalyNumber [feature_number], features.ILICAOdometer [feature_ODO], features.StationNum+mlv.FactorNum [IRAS_chainage_m],features.ILICADepthPct [depth_fraction], features.ILICALength [length_mm], features.ILICAWidth [width_mm], b.[OD_inch], b.[WT_mm], b.[grade_MPa], b.[toughness_J], format(b.[install_date],'yyyy-MM-dd') [install_date], d.MAOP_kPa, b.begin_PS, b.end_PS, d.begin_maop, d.end_maop, @begin [start_ILI_segment_m], @end [end_ILI_segment_m], mc.Algorithm [MCAlgorithm], mc.FailureProbability [MCPOE], mc.LeakProbability [MCLPOE], mc.RuptureProbability [MCRPOE] from InlineInspectionCrackAnomaly features
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILICAStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILICATypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId

        join ( select ll.id [llid], ps.BeginStationSeriesId, ps.PipeOutsideDiameter [OD_inch], ps.PipeWallThickness [WT_mm], ps.PipeGrade [grade_MPa], ps.PipeToughness [toughness_J], ps.EffectiveStartDate [install_date], ps.BeginStationNum+mlv1.FactorNum [begin_PS], ps.EndStationNum+mlv2.FactorNum [end_PS]  from PipeSegment ps 
        join StationSeries ss on ss.id = ps.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on ps.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on ps.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, ps.BeginStationSeriesId asc*/  ) b on ((features.StationNum+mlv.FactorNum) between b.begin_PS and b.end_PS) and b.llid = ll.id

        join ( select ll.id [llid], maop.BeginStationSeriesId, maop.MaxAllowablePressure [MAOP_kPa], maop.BeginStationNum+mlv1.FactorNum [begin_maop], maop.EndStationNum+mlv2.FactorNum [end_maop] from maop maop
        join StationSeries ss on ss.id = maop.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on maop.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on maop.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, maop.BeginStationSeriesId asc*/ ) d on ((features.StationNum+mlv.FactorNum) between d.[begin_maop] and d.[end_maop]) and d.llid = ll.id

        join MonteCarlo_Crack mc on features.id = mc.InlineInspectionFeatureId and mc.Algorithm like '%Crack_POF_CSCC%'

        where ll.LineName like '%'+@line+'%' and 
        features.InlineInspectionRangeId = @ILIR_id and
        ld.Code = 'Active' and
        ld2.Code in (select * from @crackcscc_t) and
        (features.StationNum+mlv.FactorNum) between @begin and @end
        order by (features.StationNum+mlv.FactorNum) asc
        ELSE IF @simulation = 'crackMD'
        select ll.linename [line], features.id [InlineInspectionCAId], ld4.code [vendor], features.ILICAGirthWeldNum [US_GWD_number], features.ILICAGirthWeldNumDS [DS_GWD_number], ld3.code [tool], format(ilir.ILIRStartDate,'yyyy-MM-dd') [ILIRStartDate], ld.code [status], ld2.code [type], features.ILICAComment [vendor_comment], features.ILICAAnomalyNumber [feature_number], features.ILICAOdometer [feature_ODO], features.StationNum+mlv.FactorNum [IRAS_chainage_m],features.ILICADepthPct [depth_fraction], features.ILICALength [length_mm], features.ILICAWidth [width_mm], b.[OD_inch], b.[WT_mm], b.[grade_MPa], format(b.[install_date],'yyyy-MM-dd') [install_date], c.[seam_toughness_J] [toughness_J], d.MAOP_kPa, b.begin_PS, b.end_PS, c.begin_VTS, c.end_VTS, d.begin_maop, d.end_maop, @begin [start_ILI_segment_m], @end [end_ILI_segment_m], mc.Algorithm [MCAlgorithm], mc.FailureProbability [MCPOE], mc.LeakProbability [MCLPOE], mc.RuptureProbability [MCRPOE]	from InlineInspectionCrackAnomaly features
        join InlineInspectionRange ilir on features.InlineInspectionRangeId = ilir.Id
        join InlineInspection ili on ilir.InlineInspectionId = ili.id
        join StationSeries ss on ilir.BeginStationSeriesId = ss.Id
        join LineLoop ll on ss.LineLoopId = ll.Id
        join ListDomain ld on features.ILICAStatusDomainId = ld.Id
        join ListDomain ld2 on features.ILICATypeDomainId = ld2.id
        join ListDomain ld3 on ilir.ILIRToolDomainId = ld3.id
        join ListDomain ld4 on ili.ILICompanyDomainId = ld4.Id
        join MLVCorrection mlv on features.StationSeriesId = mlv.StationSeriesId

        join ( select ll.id [llid], ps.BeginStationSeriesId, ps.PipeOutsideDiameter [OD_inch], ps.PipeWallThickness [WT_mm], ps.PipeGrade [grade_MPa], ps.PipeToughness [toughness_J], ps.EffectiveStartDate [install_date], ps.BeginStationNum+mlv1.FactorNum [begin_PS], ps.EndStationNum+mlv2.FactorNum [end_PS]  from PipeSegment ps 
        join StationSeries ss on ss.id = ps.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on ps.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on ps.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, ps.BeginStationSeriesId asc*/  ) b on ((features.StationNum+mlv.FactorNum) between b.begin_PS and b.end_PS) and b.llid = ll.id

        full outer join (select vts.BeginStationSeriesId, vt.VirtualTableName, (vts.BeginStationNum+mlv1.FactorNum) as [begin_VTS], (vts.EndStationNum+mlv2.FactorNum) as [end_VTS], vtdn.DataFieldValueNum as [seam_toughness_J] from VirtualTableDataNum vtdn
        join VirtualTableStationing vts on vtdn.VirtualTableStationingId = vts.VirtualTableStationingId and vtdn.VirtualTableId = vts.VirtualTableId
        join MLVCorrection mlv1 on vts.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on vts.EndStationSeriesId = mlv2.StationSeriesId
        join StationSeries ss on vts.BeginStationSeriesId = ss.id
        join VirtualTable vt on vtdn.VirtualTableId = vt.Id
        where vtdn.VirtualTableId = 121) c on ((features.StationNum+mlv.FactorNum) between c.begin_VTS and c.end_VTS) and c.beginStationSeriesId = ss.id

        join ( select ll.id [llid], maop.BeginStationSeriesId, maop.MaxAllowablePressure [MAOP_kPa], maop.BeginStationNum+mlv1.FactorNum [begin_maop], maop.EndStationNum+mlv2.FactorNum [end_maop] from maop maop
        join StationSeries ss on ss.id = maop.BeginStationSeriesId
        join LineLoop ll on ss.LineLoopId = ll.Id
        join MLVCorrection mlv1 on maop.BeginStationSeriesId = mlv1.StationSeriesId
        join MLVCorrection mlv2 on maop.EndStationSeriesId = mlv2.StationSeriesId
        --where ll.LineName like @line
        /*order by ll.id asc, maop.BeginStationSeriesId asc*/ ) d on ((features.StationNum+mlv.FactorNum) between d.[begin_maop] and d.[end_maop]) and d.llid = ll.id

        join MonteCarlo_Crack mc on features.id = mc.InlineInspectionFeatureId and mc.Algorithm like '%Crack_POF_MD%'

        where ll.LineName like '%'+@line+'%' and 
        features.InlineInspectionRangeId = @ILIR_id and
        ld.Code = 'Active' and
        ld2.Code in (select * from @crackmd_t) and
        (features.StationNum+mlv.FactorNum) between @begin and @end
        order by (features.StationNum+mlv.FactorNum) asc
        """
    
    df = pd.read_sql_query(q2, conn)
    conn.close()
    return df

##conn.close()

if __name__ == '__main__':
    sttime = time.time()
    
    #main_path = r'Z:\Plains Midstream\2019_01_Annual Risk Data Support\3_Engineering\Results & QC\3 Risk Drivers\20190621-415p-system_STAGE'
    #threat_path = r''
    #folder_path = main_path + threat_path

    CSV_PATH = 'reportable_dyn_seg.csv'
    CSV_PATH = abspath(join(dirname(__file__), CSV_PATH))
    
    data = pd.read_csv(CSV_PATH)
    #os.chdir(main_path)
    
    linename = data['Line Name'].values
    begins = data['Begin Measure (m)'].values
    ends = data['End Measure (m)'].values
    mids = (begins+ends)/2
    extents = data['Length (m)'].values
    prob_matrix_placement = data ["Matrix 'Probability' Label"].values
    consq_matrix_placement = data["Matrix 'Consequence' Label"].values
    risk_matrix_placement = data['Legend Label'].values
    box_matrix_placement = data['Cell Label'].values

    ILI_segments, ILI_seg_len, line_len= get_ILI_segs(linename[0])
    begins_normed = normalize_to_ili_seg(begins,ILI_segments)
    mids_normed = normalize_to_ili_seg(mids,ILI_segments)
    
    ILIR_ID = get_ILIR(linename[0])

    master_df = pd.DataFrame()

###########METHOD 1: Use this one for checking each dynamic segment in query --corrosion | NCD | crackSCC | crackMD | crackCSCC
##########    for pipe, start, length, pmp, cmp, rmp, bmp in zip(linename, begins_normed, extents, prob_matrix_placement, consq_matrix_placement, risk_matrix_placement, box_matrix_placement):
############        print(pipe,start,length,ILIR_ID,model, sep='--')
##########        temp = get_features_between_range(pipe,start,length,ILIR_ID,'corrosion',surface='E')
##########        temp['LIKELIHOOD'] = pmp
##########        temp['CONSEQUENCE'] = cmp
##########        temp['RISK'] = rmp
##########        temp['BOX'] = bmp
##########        master_df = master_df.append(temp)
##########        del temp

###########METHOD 2: Use this one for checking normalized dynamic segments to ILI segments --corrosion | NCD | crackSCC | crackMD | crackCSCC
    for pipe, start, pmp, cmp, rmp, bmp in zip(linename, mids_normed, prob_matrix_placement, consq_matrix_placement, risk_matrix_placement, box_matrix_placement):
##        print(pipe,start,length,ILIR_ID,model, sep='--')
        temp = get_features_between_range(pipe,start,ILI_seg_len,ILIR_ID,'crackMD',surface='E')
        temp['LIKELIHOOD'] = pmp
        temp['CONSEQUENCE'] = cmp
        temp['RISK'] = rmp
        temp['BOX'] = bmp
        master_df = master_df.append(temp)
        del temp

    master_df.reset_index(drop=True, inplace=True)

    file_path = linename[0].replace(' ','-') + '.csv'
    
    print(master_df)
    print('Time until completion: {} seconds.'.format(time.time()-sttime))
    #os.chdir(folder_path)
    #master_df.to_csv(file_path)
    master_df.to_clipboard()
    print('Exported.')

