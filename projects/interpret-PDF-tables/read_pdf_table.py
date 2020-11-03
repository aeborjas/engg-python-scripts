import camelot
import os
#following example from https://camelot-py.readthedocs.io/en/master/

# tables = camelot.read_pdf(r'Calculators\projects\interpret-PDF-tables\2019 Rangeland NPS 8 North Ferrier Cut Outs - Project Summary.pdf')
tables = camelot.read_pdf('projects\\interpret-PDF-tables\\ipl_algorithm.pdf')

tables.export(os.path.join(os.path.dirname(__file__),'algo_tables.xlsx'), f='xlsx', compress=True)