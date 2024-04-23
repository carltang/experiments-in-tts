import numpy as np
import os
import codecs

def GetFileList(dir, fileList, postfix):
	if os.path.isfile(dir):
		if dir.find("." + postfix) >= 0:
			fileList.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			GetFileList(newDir, fileList, postfix)
	return fileList

npz_list = GetFileList("tfcc", [], 'npz')

npzdict = {}

key_list = ["conv2d_1/kernel",
			"conv2d_1/bias",
			"dense_1/kernel",
			"dense_1/bias",
			"dense_2/kernel",
			"dense_2/bias"]

recodeFile = codecs.open("polySet.txt", "w", "gb18030")

for npz_file in npz_list:
	hanzi = npz_file[-5]
	dictFile = "../dnn_train_data_word_pos/" + hanzi + ".dict"
	print(hanzi)
	recodeFile.write("{}".format(hanzi))
	dict = codecs.open(dictFile, "r", "gb18030")
	lines = dict.readlines()
	for line in lines:
		parts = line.split()
		recodeFile.write(" {}".format(parts[0]))
	recodeFile.write("\n")
	dict.close()
	data = np.load(npz_file)
	for key in key_list:
		npzdict[hanzi + "/" + key] = data[key]
	data.close()

np.savez("polyphone.npz", **npzdict)
recodeFile.close()
