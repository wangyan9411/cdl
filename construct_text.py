import csv
from data import read_user
import numpy as np
p = 4 
user_id = 1
# read predicted results
dir_save = 'cdl%d' % p
csvReader = csv.reader(open('raw-data.csv','rb'))
d_id_title = dict()
c = ''
for i,row in enumerate(csvReader):
    if i == 0:
        continue
    abastract = row[4]
    c += ' '.join(row[4].lower().split()) + '\n'

open('raw-text', 'w').write(c)
    
    
