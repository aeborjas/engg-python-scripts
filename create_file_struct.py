import glob
import os

folder = r'Z:\Plains Midstream\2019_01_Annual Risk Data Support\3_Engineering\Results & QC\3 Risk Drivers'
run = r'\20190226-3p-system'
consq = r'\Safety'
threat = r'\RES'

src_path = folder + run + consq + threat

output_folder = r'Z:\Plains Midstream\2019_01_Annual Risk Data Support\3_Engineering\Results & QC\3 Risk Drivers\Feature Deactivation Exercise'
output_consq = r'\Safety'

output_path = output_folder + output_consq

os.chdir(src_path)

threats = ['EC','IC','SCC','MD','CSCC','RES']

#print(glob.glob(output_path+"\\*",recursive=True))  
#print(os.listdir(output_path))

for i,x in enumerate(glob.glob("*.csv")):
    if x[:-4] not in os.listdir(output_path):
        print(x,"not in output")
        temp_path = output_path + r"\\" + x[:-4]
        os.mkdir(temp_path)

        for y in threats:
            os.mkdir(temp_path + r"\\" + y)

        print(i,'done.\n')


#for i, x in enumerate(glob.glob("*.csv")):
    
#    temp_path = output_path + r"\\" + x[:-4]
#    os.mkdir(temp_path)

#    for y in threats:
#        os.mkdir(temp_path + r"\\" + y)

#    print(i,'done.\n')