from pdfrw import PdfReader
import os

##path = 'O:\\Engineering\\Engineering Standards and Papers\\International Pipeline Conference (IPC) 1996'
path = 'O:\\Engineering\\Engineering Standards and Papers\\International Pipeline Conference (IPC) '
##years = ['1996',
##         '1998',
##         '2000',
##         '2002',
##         '2004',
##         '2006',
##         '2008',
##         '2010',
##         '2012',
##         '2014',
##         '2016']
years = ['2002']

for year in years:
    pathi = path+year
    print(pathi,'-----------------------------------------------------------------')
    for root, dirs, files in os.walk(pathi):
        for file_ in files:
            ext = os.path.splitext(file_)[-1].lower()
            if ext == '.pdf':
    ##            print(file_)
                reader = PdfReader(os.path.join(root,file_))
                print(reader.Info.Title,'|',file_)
