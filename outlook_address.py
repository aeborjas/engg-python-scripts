##import win32com.client
##
##DEBUG = 0
##
##class MSOutlook:
##    def __init__(self):
##        self.outlookFound = 0
##        try:
##            self.oOutlookApp = \
##                win32com.client.gencache.EnsureDispatch("Outlook.Application")
##            self.outlookFound = 1
##        except:
##            print("MSOutlook: unable to load Outlook")
##        
##        self.records = []
##
##
##    def loadContacts(self, keys=None):
##        if not self.outlookFound:
##            return
##
##        # this should use more try/except blocks or nested blocks
##        onMAPI = self.oOutlookApp.GetNamespace("MAPI")
##        ofContacts = \
##            onMAPI.GetDefaultFolder(win32com.client.constants.olFolderContacts)
##
##        if DEBUG:
##            print("number of contacts:", len(ofContacts.Items))
##
##        for oc in range(len(ofContacts.Items)):
##            contact = ofContacts.Items.Item(oc + 1)
##            if contact.Class == win32com.client.constants.olContact:
##                if keys is None:
##                    # if we were't give a set of keys to use
##                    # then build up a list of keys that we will be
##                    # able to process
##                    # I didn't include fields of type time, though
##                    # those could probably be interpreted
##                    keys = []
##                    for key in contact._prop_map_get_:
##                        if isinstance(getattr(contact, key), (int, str, unicode)):
##                            keys.append(key)
##                    if DEBUG:
##                        keys.sort()
##                        print("Fields\n======================================")
##                        for key in keys:
##                            print(key)
##                record = {}
##                for key in keys:
##                    record[key] = getattr(contact, key)
##                if DEBUG:
##                    print(oc, record['FullName'])
##                self.records.append(record)
##
##
##if __name__ == '__main__':
##    if DEBUG:
##        print("attempting to load Outlook")
##    oOutlook = MSOutlook()
##    # delayed check for Outlook on win32 box
##    if not oOutlook.outlookFound:
##        print("Outlook not found")
##        sys.exit(1)
##
##    fields = ['FullName',
##                'CompanyName', 
##                'MailingAddressStreet',
##                'MailingAddressCity', 
##                'MailingAddressState', 
##                'MailingAddressPostalCode',
##                'HomeTelephoneNumber', 
##                'BusinessTelephoneNumber', 
##                'MobileTelephoneNumber',
##                'Email1Address',
##                'Body'
##                ]
##
##    if DEBUG:
##        import time
##        print("loading records...")
##        startTime = time.time()
##    # you can either get all of the data fields
##    # or just a specific set of fields which is much faster
##    #oOutlook.loadContacts()
##    oOutlook.loadContacts(fields)
##    if DEBUG:
##        print("loading took %f seconds" % (time.time() - startTime))
##
##    print("Number of contacts: %d" % len(oOutlook.records))
##    print("Contact: %s" % oOutlook.records[0]['FullName'])
##    print("Body:\n%s" % oOutlook.records[0]['Body'])

#############################
####
####import win32com.client
####import csv
####from datetime import datetime
####
##### Outlook
####outApp = win32com.client.gencache.EnsureDispatch("Outlook.Application")
####outGAL = outApp.Session.GetGlobalAddressList()
####entries = outGAL.AddressEntries
####
##### Create a dateID
####date_id = (datetime.today()).strftime('%Y%m%d')
####
##### Create empty list to store results
####data_set = list()
####
##### Iterate through Outlook address entries
####for entry in entries:
####    if entry.Type == "EX":
####        user = entry.GetExchangeUser()
####        if user is not None:
####            if len(user.FirstName) > 0 and len(user.LastName) > 0:
####                row = list()
####                row.append(date_id)
####                row.append(user.Name)
####                row.append(user.FirstName)
####                row.append(user.LastName)
####                row.append(user.JobTitle)
####                row.append(user.City)
####                row.append(user.PrimarySmtpAddress)
####                try:
####                    row.append(entry.PropertyAccessor.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x3a26001e"))
####                except:
####                    row.append('None')
####                
####                # Store the user details in data_set
####                data_set.append(row)
####
##### Print out the result to a csv with headers
####with open(date_id + 'outlookGALresults.csv', 'w', newline='', encoding='utf-8') as csv_file:
####    headers = ['DateID', 'DisplayName', 'FirstName', 'LastName', 'JobTitle', 'City', 'PrimarySmtp', 'Country']
####    wr = csv.writer(csv_file, delimiter=',')
####    wr.writerow(headers)
####    for line in data_set:
####        wr.writerow(line)


##################################################
        ## GET CONTACTS
import os, sys
import win32com.client


constants = win32com.client.constants
outApp = win32com.client.gencache.EnsureDispatch("Outlook.Application")
ns = outApp.GetNamespace("MAPI")

def items(contacts):
    items = contacts.Items
    item = items.GetFirst()
    while item:
        yield item
        item = items.GetNext()


for contact in items(ns.GetDefaultFolder(constants.olFolderContacts)):
    if contact.Class == constants.olContact:
        print(contact)
    for field in ['FullName','CompanyName','Email1Address']:
        print("",field,getattr(contact,field,"<Unknown>"))
