	 # coding=utf-8
from __future__ import print_function
import codecs
import re
import os
import sys
import numpy as np
#import tensorflow as tf
import gensim
#from keras.utils import np_utils

W2V = None
P2V = None
GRAPH = None
PY2Y = {}

W2V_LEN = 100
HALF_WINDOW_LEN = 7

#载入w2v
def loadw2v(datapath):
	global W2V
	w2vpath = datapath + "/dim100_char.w2v.bin"
	W2V = gensim.models.KeyedVectors.load_word2vec_format(w2vpath, binary=True, unicode_errors='ignore')

def genPY2Y(label):
	global PY2Y
	fplabel = codecs.open(label, "r", "gb18030")
	y = 0
	PY2Y = {}
	while True:
		line = fplabel.readline()
		if not line:
			break
		while line.find("{") >= 0:
			begin = line.find("{")
			end = line.find("}")
			label = line[begin + 1:end]
			if not (label in PY2Y):
				PY2Y[label] = y
				y += 1
			line = line[:begin] + line[end + 1:]
	fplabel.close()

def savePY2Y(filename):
	with open(filename, 'w') as f:
		for key,value in PY2Y.items():
			f.write("{} {}\n".format(key, value))


def trans_to_dnn_train(hanzi, label, outputDir):
	X_featureList = []
	Y_featureList = []
	fplabel = codecs.open(label, "r", "gb18030", errors="ignore")
	while True:
		#带拼音
		withlabel = fplabel.readline().strip()

		if not withlabel:
			break

		labels = []
		indexes = []

		print ("process: ", withlabel)
		start_pos = 0
		index = withlabel.find(hanzi, start_pos)
		while index >= 0:
			if index < len(withlabel) - 1 and withlabel[index + 1] == "{":
				end = withlabel.find("}", index)
				if (end < 0):
					print ("can not find } in {}".format(withlabel))
					exit(1)
				label = withlabel[index + 2: end]
				Y = PY2Y[label]
				labels.append(Y)
				indexes.append(index)
				withlabel = withlabel[:index+1] + withlabel[end+1:]
				start_pos = index + 1

				index = withlabel.find(hanzi, start_pos)
			else:
				start_pos = index + 1
				index = withlabel.find(hanzi, start_pos)
			#else:
				#print("err !!")
				#exit()
				#labels.append(-1)
				#withlabel = withlabel[:index] + withlabel[index+1:]

		print ("without label: ", withlabel)
		#cursentence构建完成, 下一步构建模型输入向量

		for index in range(len(indexes)):
			i = indexes[index]
			show_info = ""
			X = np.zeros((HALF_WINDOW_LEN * 2 + 1, W2V_LEN), dtype=np.float32)
			for j in range(0, HALF_WINDOW_LEN * 2 + 1):
				if i + j - HALF_WINDOW_LEN >= 0 and i + j - HALF_WINDOW_LEN < len(withlabel):
					show_info += withlabel[i + j - HALF_WINDOW_LEN]
					try:
						w_id = W2V[withlabel[i + j - HALF_WINDOW_LEN]]
					except:
						#print("can not find {} in w2v".format(withlabel[i+j-5]))
						w_id = W2V['*']
					X[j] = w_id
				else:
					show_info += '*'
			X_featureList.append(X)
			Y_featureList.append(labels[index])
			show_info += " " + str(labels[index])
			print(show_info)
	fplabel.close()

	x_array = np.array(X_featureList)
	y_array = np.array(Y_featureList)
	print (x_array.shape)
	print (y_array.shape)
	np.save(outputDir + "/" + hanzi + "_x.bin", x_array)
	np.save(outputDir + "/" + hanzi + "_y.bin", y_array)

def GetFileList(dir, fileList):
	if os.path.isfile(dir):
		if dir.find(".txt") >= 0:
			fileList.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			GetFileList(newDir, fileList)
	return fileList

def del_dir(path):
	if os.path.exists(path):
		ls = os.listdir(path)
		for i in ls:
			c_path = os.path.join(path, i)
			if os.path.isdir(c_path):
				del_file(c_path)
			else:
				os.remove(c_path)
		os.rmdir(path)



def genData(dataDir):
	# 使用frontend运行train/test文件夹下面所有的withoutlabel文件
	loadw2v("w2v")

	#f = codecs.open("pinyin2label.txt", "r", "gb18030")

	del_dir("dnn_train_data_char")
	os.system("mkdir dnn_train_data_char")

	# 先转换training data
	trainInputList = GetFileList("{}/with_label".format(dataDir), [])

	for inputFile in trainInputList:
		if inputFile.find("_test") >= 0:
			hanzi = inputFile[-10]
		else:
			hanzi = inputFile[-5]
		outDir = "dnn_train_data_char/"
		outFile = outDir + hanzi + "_x.bin.npy"
		if not os.path.exists(outFile):
			genPY2Y(inputFile)
			savePY2Y(outDir + hanzi + ".dict")
			print (PY2Y)
			trans_to_dnn_train(hanzi, inputFile, outDir)


genData(sys.argv[1])