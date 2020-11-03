import numpy as np
import pandas as pd

input('Copy dataset into dataframe:')
df = pd.read_clipboard()

results = dict(
            line=np.array([]),
            ch=np.array([]),
            stn=np.array([])
            )

results = pd.DataFrame(results)

for x in df.iterrows():
    stn = np.arange(x[1]['start (m)'],x[1]['end (m)'],x[1]['corrected segment length'])
    ch = np.arange(0.0,x[1]['length (m)'],x[1]['corrected segment length'])
    temp_size = stn.size
    line = np.repeat(x[1]['line'],temp_size)

    temp_df = dict(
                line=line,
                ch=ch,
                stn=stn)
    temp_df = pd.DataFrame(temp_df)
    results = results.append(temp_df)

print(results)
##df2 = pd.DataFrame(temp)
##df2.to_clipboard()
