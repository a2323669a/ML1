import pandas
import numpy as np
import csv

w = [-0.03053024 ,-0.02277121  ,0.20342201, -0.22243108, -0.05347607, 0.50973682,
 -0.55534354 , 0.00349552, 1.0865513 ]

b = 1.742024455315198

df_data = pandas.read_csv("./data/PM2.5/test.csv",header=None,encoding='Big5')

df_data = df_data[df_data[1] == 'PM2.5']

data_x = df_data.iloc[:,2:].values.astype(np.float)
df_head = df_data.iloc[:,0].values.astype(np.str).tolist()

f = open('./result/PM2.5/test.csv',"w",newline='')
writer = csv.writer(f)
writer.writerow(['id','value'])

for i in range(len(data_x)):
    x = data_x[i]
    y = np.sum(np.multiply(w,x)) + b
    writer.writerow([df_head[i],y])

f.close()