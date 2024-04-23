import os
import sys
from single_train_h5 import do_train
import numpy as np

def GetFileList(dir, fileList, postfix):
	if os.path.isfile(dir):
		if dir.find("." + postfix) >= 0:
			fileList.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			GetFileList(newDir, fileList, postfix)
	return fileList

if os.path.exists("train_results.txt"):
	os.remove("train_results.txt")

#路径参数
input_path = 'dnn_train_data_word_pos'

#最后训练输入0，测试用1
type = int(sys.argv[1])

if type == 1:
	val_acc = {}
	acc_len = 0

h5_list = GetFileList(input_path, [], 'dict')
for dict_file in h5_list:
	hanzi = dict_file[-6]
	if type == 0:
		os.system("python single_train_h5.py {}".format(hanzi))
	else:
		val_acc[hanzi] = np.array(do_train(hanzi))
		acc_len = val_acc[hanzi].shape[0]

if type == 1:
	print(val_acc)
	av = np.zeros(acc_len)
	for value in val_acc.values():
		av += value
	av /= len(val_acc)
	print ("{}".format(av))
