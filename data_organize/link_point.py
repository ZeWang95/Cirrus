import json
import numpy as np 
import os


with open('/data/dataset/volvo/volvo_june_2019/code/name_dict.json') as f:
	data = json.load(f)

files = os.listdir('dataset/raw_data/lidar')

cc = 0
for f in files:
	if f in data.keys():
		source = os.path.join('/data/dataset/volvo/volvo_june_2019', data[f])
		target = os.path.join('/data/dataset/volvo/data_all/dataset/raw_data/lidar_u', f)
		os.system('ln -s %s %s' %(source, target))