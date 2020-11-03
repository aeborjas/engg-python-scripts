import win32com.client, datetime, pprint, pandas as pd, numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s","--start",help="start date in YYYY-MM-DD format.")
parser.add_argument("-d","--days", help="number of days to gather data.")
parser.add_argument("--group",help="flag to allow grouping of data result")
parser.add_argument("--filter",help="filter flags for data")
args = parser.parse_args()

today = datetime.datetime.today()

def getCalendarEntries(start, days=1):
    """
    Returns calender entries for days default is 1
    """
    Outlook = win32com.client.Dispatch("Outlook.Application")
    ns = Outlook.GetNamespace("MAPI")
    # appointments = ns.GetDefaultFolder(9).Items
    appointments = ns.Folders['Armando_Borjas@dynamicrisk.net'].Folders['Calendar'].Folders['Timesheet Tasks'].Items
    appointments.Sort("[Start]")
    appointments.IncludeRecurrences = "True"
    
    begin = datetime.datetime.strptime(start,"%Y-%m-%d")
    tomorrow= begin + datetime.timedelta(int(days))
    end = tomorrow.date().strftime("%Y-%m-%d")
    appointments = appointments.Restrict("[Start] >= '" +start+ "' AND [END] <= '" +end+ "'")
    events={'start':[],'subject':[],'description':[],'duration':[]}
    for a in appointments:
        adate=datetime.datetime.fromtimestamp(timestamp=a.Start.timestamp(), tz=a.Start.tzinfo)
        events['start'].append(adate)
        events['subject'].append(a.Subject)
        events['description'].append(a.Body)
        events['duration'].append(a.Duration)

    events = pd.DataFrame(events)#.astype(convert_dict)
    events['start'] = events['start'].dt.tz_convert(None)
    events.set_index('start', inplace=True)
    return events

def addevent(start, duration, subject):
    import win32com.client
    oOutlook = win32com.client.Dispatch("Outlook.Application")
    ns = oOutlook.GetNamespace("MAPI")
    tsht_folder = ns.Folders['Armando_Borjas@dynamicrisk.net'].Folders['Calendar'].Folders['Timesheet Tasks'].Items
    tsht_items = tsht_folder.Restrict("[Start] >= '02/26/2019' AND [END] <= '02/27/2019'")
    new_entry = tsht_items.Add()
    new_entry.Start = start
    new_entry.Duration = duration
    new_entry.Subject = subject
    new_entry.Save()
    #appointment = oOutlook.CreateItem(1) # 1=outlook appointment item
    #appointment.Start = start
    #appointment.Subject = subject
    #appointment.Duration = 30
    #appointment.Location = 'Sprortground'
    #appointment.ReminderSet = True
    #appointment.ReminderMinutesBeforeStart = 1
    #appointment.Save()
    return


if __name__ == '__main__':
    if not args.group: 
        pprint.pprint(getCalendarEntries(args.start,days=args.days))
    elif args.group:
        x = getCalendarEntries(args.start,days=args.days)
        pprint.pprint(x.groupby(args.filter).sum().duration/60.0)
        pprint.pprint(x.groupby(args.filter).sum().duration.sum()/60.0)
        # pprint.pprint(x.loc[lambda y: y.subject==str(args.filter),"description"]/60.0)
    # print(args)

