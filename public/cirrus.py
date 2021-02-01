import os, sys, pdb
import numpy as np 
import argparse

class Cirrus(object):
	def __init__(self, path):
		self.path = os.path.abspath(path)

	def _target_path(self, *args):
		return os.path.join(self.path, *args)

	def _cp_data(self, source, target, soft_link=True):
		if soft_link:
			cmd = 'ln -s'
		else:
			cmd = 'cp'
		os.system('%s %s %s' %(cmd, source, target))

	def creat_joint_dataset(self):
		for i in range(1, 8):
			assert os.path.exists(self._target_path('volvo_batch_%d_dataset' %i)), 'Mising directory volvo_batch_%d_dataset.'%i

		os.makedirs(self._target_path('data_all'), exist_ok=True)
		os.makedirs(self._target_path('data_all', 'raw_data'), exist_ok=True)
		os.makedirs(self._target_path('data_all', 'raw_data', 'lidar'), exist_ok=True)
		os.makedirs(self._target_path('data_all', 'json_files'), exist_ok=True)

		file_list = ['json_files', 'raw_data/lidar']

		for i in range(1, 8):
			for f in file_list:
				self._cp_data(self._target_path('volvo_batch_%d_dataset' %i, 'dataset', f, '*'), 
					self._target_path('data_all', f), True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./', help='Path to the dataset.')


    args = parser.parse_args()
    cirrus_dataset = Cirrus(args.path)
    cirrus_dataset.creat_joint_dataset()