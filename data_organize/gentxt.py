import os 
import numpy as np 
import json
import pdb

def load_json(path):
	f = open(path)
	data = json.load(f)
	f.close()
	return data['labels']

def decode_quan(q):
	dd = np.arctan(2*(q[0]*q[1] + q[2]*q[3])/ ((1-2*(q[1]**2+q[3]**2))+1e-8))
	return dd

def gen_label_list(anno):
	list_out = []
	# pdb.set_trace()
	for an in anno:
		p = an['position']
		q = an['quaternion']
		s = an['size']
		q = decode_quan(q)
		# out = np.concatenate([p, s, q], 0)
		out = np.vstack([s[2], s[0], s[1], -p[1], -(p[2] - s[2]/2.0), p[0], q])
		list_out.append(out)
		# pdb.set_trace()
		# print(an['label_class'])
	return np.array(list_out)#, an['label_class']

if __name__ == '__main__':

	root_dir = '../dataset/json_files'
	txt_dir = '../dataset/txt_files'

	ll = os.listdir(root_dir)
	ll.sort()

	for i in ll:
		label_file = i

		txt_file = label_file.replace('.json', '.txt')

		target_file = open(os.path.join(txt_dir, txt_file), 'w')

		anno_file = load_json(os.path.join(root_dir, label_file))
		objs = gen_label_list(anno_file)
		# print objs.shape
		label = np.array(objs, np.float32)

		cls_list = [a['label_class'] for a in anno_file]
		# pdb.set_trace()
		for j in range(len(cls_list)):

			obj = objs[j]

			out_content = cls_list[j]
			out_content += ' '

			for k in range(7):
				out_content += '0.0 '

			for k in range(7):
				out_content += str(obj[k][0])
				if k != 6:
					out_content += ' '
			out_content += '\n'

			target_file.write(out_content)

		target_file.close()