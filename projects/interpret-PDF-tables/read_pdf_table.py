import camelot
import os
#following example from https://camelot-py.readthedocs.io/en/master/

tables = camelot.read_pdf(r'Cut Outs - Project Summary.pdf')

tables.export(os.path.join(os.path.dirname(__file__),'algo_tables.xlsx'), f='xlsx', compress=True)