import json
import numpy as np
import os, pdb

gau_list = ['2019-05-17-volvo-gaussian', '2019-05-20_volvo-gaussian']

dic = {}

for g in gau_list:
	path = os.path.join('..', g)
	folder_list = os.listdir(path)

	for f in folder_list:
		files = os.listdir(os.path.join(path, f, 'luminar_full_cloud'))
		files = [f for f in files if f.endswith('.xyz')]
		files.sort()

		uni_files = os.listdir(os.path.join(path.replace('gaussian', 'uniform'), f, 'luminar_full_cloud'))
		uni_files = [f for f in uni_files if f.endswith('.xyz')]
		uni_files.sort()

		print(len(files), len(uni_files))

		size = min(len(files), len(uni_files))
		# print(size)
		# pdb.set_trace()
		for i in range(size):
			dic[files[i]] = os.path.join(path.replace('gaussian', 'uniform'), f, 'luminar_full_cloud', uni_files[i])[3:]

f = open('name_dict.json', 'w')
json.dump(dic, f)
f.close()