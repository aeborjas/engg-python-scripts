import pandas as pd
from os.path import abspath, dirname, join

CSV_FILE_NAME = 'table_segments.csv'
CSV_FILE_NAME = abspath(join(dirname(__file__), CSV_FILE_NAME))

pd.set_option("display.max_columns", 50)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', None)

df = pd.read_csv(CSV_FILE_NAME)

grouped_df = df.groupby('tablename').BeginStationNum.agg(mean=lambda x: x.diff().mean(),
                                                    min=lambda x: x.diff().min(),
                                                    p25th=lambda x: x.diff().quantile(0.25),
                                                    p50th=lambda x: x.diff().quantile(0.50),
                                                    p75th=lambda x: x.diff().quantile(0.75),
                                                    max=lambda x: x.diff().max())

grouped_df.to_clipboard()
print(grouped_df)