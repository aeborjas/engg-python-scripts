import pandas as pd
import glob
import re
import os
import time

#record start time
start = time.time()

#link of stored excel files to merge
link = r'Z:\Plains Midstream\2018_01_IRAS_Implementation\3_Engineering\Quantitative Risk Run (V6)\Results & QC\SCC Method 1 and Method 2 Comparison Analysis'
os.chdir(link)

#grabbing only .xlsx files
glob_link = link+r'\*.xlsx'

excel_names = glob.glob(glob_link)

#regular expression substituting links for names
excel_names = [re.sub(r'(.*)(\\)(.*\.xlsx)','\g<3>',link) for link in excel_names]

#converting excel files to DataFrames
print("Loading Excel files...")
excels = [pd.ExcelFile(name) for name in excel_names]
print("Loading complete.")

#parsing data
print("Parsing Excel files...")
frames = [x.parse(x.sheet_names[0], header=None, index_col=None) for x in excels]
print("Parsing complete.")

#Remove the first columns of all but the first file
print("Removing first 3 rows off Excel files other than first...")
frames[1:] = [df[3:] for df in frames[1:]]
print("Complete.")

#concatenate the data
print("Concatenating...")
combined = pd.concat(frames)
print("Complete.")

#output the file
print("Outputting...")
combined.to_excel("combined_"+excel_names[0], header=False, index=False)

#record end time
end = time.time()

print("Took {} seconds".format(end-start))
