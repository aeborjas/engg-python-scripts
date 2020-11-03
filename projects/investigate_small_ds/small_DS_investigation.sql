declare @line varchar(255) = 'LS7108'

lineloop:
select * from LineLoop ll
where ll.LineName = @line 

Stationseries:
select
ll.linename,
ss.id, 
ss.BeginStationNum, 
ss.EndStationNum, 
'STATIONSERIES' [tablename] 
from StationSeries ss
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	


select ll.LineName, 
t.BeginStationSeriesId, 
t.BeginStationNum, 
t.EndStationSeriesId, 
t.EndStationNum, 
'CLASSAREA' [tablename]
 from ClassArea t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc	

union

select 
ll.LineName, 
t.BeginStationSeriesId, 
t.BeginStationNum, 
t.EndStationSeriesId,
t.EndStationNum,
'MAOP' [tablename]
from maop t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line		
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc	

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'PIPESEGMENT' [tablename]
from pipesegment t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc	

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'OPERATINGDATA' [tablename]
from OperatingData t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc	

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'LANDUSE' [tablename]
from LandUse t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'COVERDEPTH' [tablename]
from CoverDepth t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'GIRTHWELD' [tablename]
from GirthWeld t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'VALVE' [tablename]
from Valve t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'ELEVATION' [tablename]
from ElevationProfile t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'INCIDNT' [tablename]
from Incident t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'GEOHAZARD' [tablename]
from IRASCustomData1 t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'HYDROHAZARD' [tablename]
from IRASCustomData2 t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
'EnvironmentImpact' [tablename]
from IRASCustomData11 t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
where ll.LineName = @line	
--order by t.BeginStationSeriesId asc, t.BeginStationNum asc

union

select 
ll.LineName,
t.BeginStationSeriesId,
t.BeginStationNum,
t.EndStationSeriesId,
t.EndStationNum,
vt.VirtualTableName [tablename] from VirtualTableStationing t
join StationSeries ss on t.BeginStationSeriesId = ss.Id
join LineLoop ll on ss.LineLoopId = ll.Id
join VirtualTable vt on t.VirtualTableId = vt.Id
where ll.LineName = @line
and vt.VirtualTableName in (select * from (values 
('StructureValueSME'),
('HabitatWildlifeSensitivityCat'),
('EA_SCCSeverity'),
('EA_LFERWSeverity'),
('MDPublicAwareness'),
('MDSignage'),
('MDBuriedMarkers'),
('MDThirdPartyNotification'),
('MDPatrolFrequency'),
('MDOperatorResponse'),
('MDPipeFinding'),
('MDPipeMarking'),
('SAF_PopulationIIZ'),
('SAF_PopulationPIR'),
('SAF_PopulationEPZ'),
('EnvironmentMedium'),
('CGA_ECGrowthRateMean'),
('CGA_ECGrowthRateSD'),
('CGA_ICGrowthRateMean'),
('CGA_ICGrowthRateSD'),
('SCCCrackGrowthRate'),
('LFERWCrackGrowthRate'),
('PipelineSystem_PipelineSystemName'),
('FIN_OutflowVolume'),
('OUT_Outage_Impact_Consequence_Level')
) as t(c))
order by tablename asc, t.BeginStationSeriesId asc, t.BeginStationNum asc

--Id	VirtualTableName
--170	CGA_ECGrowthRateSD
--171	CGA_ICGrowthRateMean
--172	CGA_ICGrowthRateSD