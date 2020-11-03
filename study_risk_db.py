import sqlite3
import pandas as pd
import os
import time

os.chdir(r'N:\Python\20190328-415p-system')


#def timer_func(funct):
#    def wrapper(funct):
#        s = time.time()
#        funct()
#        print(f'Function took {time.time() - s} seconds.')
#        return 
#    return


print('Initiating connection...')
conn = sqlite3.connect('20190328-415p-system_RiskRunResults.db')
print('Connection established!')

q = """
    select * from RiskResults
    """

df = pd.read_sql_query(q, conn)

print('Query completed!')

def generate_matrix(data,threat,consequence, consq_bin='Total', by='count'):
    pd.set_option('display.max_columns',20)
    t_bins = [-float('inf'), 1e-3, 1e-2, 1e-1, 9e-1, float('inf')]
    t_lab = ['A','B','C','D','E']
    c_lab = ['Minor','Moderate','Problematic','Critical','Catastrohpic']
    if consq_bin == 'Safety':
        c_bins = [-float('inf'), 0.01, 0.1, 1.0, 10.0, float('inf')]
    elif consq_bin == 'Environment':
        c_bins = [-float('inf'), 1.0, 10.0, 100.0, 500.0, float('inf')]
    elif consq_bin == 'Economic Loss':
        c_bins = [-float('inf'), 0.1, 1.0, 10.0, 100.0, float('inf')]
    elif consq_bin == 'Total':
        c_bins = [-float('inf'), 0.1, 1., 10., 100., 500., float('inf')]
        c_lab = ['Minor', 'Moderate','Problematic','Critical','Cat1','Cat2']
    else:
        print('Consequence bin option not acceptable. Please select between Total, Safety, Environment, or Economic Loss')
        generate_matrix(threat,consequence,consq_bin)

    threat_binned = pd.cut(data[threat], t_bins, include_lowest=True, right=False, labels = t_lab)
    consequence_binned = pd.cut(data[consequence], c_bins, include_lowest=True, right=False, labels = c_lab)

    if by =='count':
        data_matrix = df.groupby([threat_binned, consequence_binned])[threat].agg('count')
        data_matrix = data_matrix.unstack().sort_index(ascending=False)
    elif by == 'length':
        data['length'] = data.EndMeasure-data.BeginMeasure
        data_matrix = df.groupby([threat_binned, consequence_binned])[threat].agg({threat:'count','length':'sum'})
        data_matrix = data_matrix.unstack().sort_index(ascending=False)
    else:
        print('Generate Matrix by count of dynamic segments or sum of segment lengths only.')
        generate_matrix(threat,consequence,consq_bin)

    return data_matrix