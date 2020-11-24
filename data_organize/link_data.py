import numpy as np 
import os
import pdb

target_path = '/data/dataset/volvo/data_all/dataset'

for i in range(1, 8):
	path = os.path.join('/data/dataset/volvo', 'volvo_batch_%d_dataset' %(i), 'dataset')

	rgbs =  os.listdir(os.path.join(path, os.path.join(target_path, 'raw_data/camera')))
	rgbs.sort()
	pcs = os.listdir(os.path.join(path, os.path.join(target_path, 'raw_data/lidar')))
	pcs.sort()
	for k in range(min(len(pcs), len(rgbs))):
		os.system('ln -s %s %s' %(os.path.join(path, 'raw_data/camera', rgbs[k]), os.path.join(target_path, 'raw_data/camera/'+pcs[k].replace('xyz', 'jpg'))))

