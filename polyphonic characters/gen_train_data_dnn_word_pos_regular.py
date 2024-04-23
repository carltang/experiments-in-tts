	 # coding=utf-8
from __future__ import print_function
import codecs
import re
import os
import sys
import numpy as np
import string
#import tensorflow as tf
import gensim
#from keras.utils import np_utils

W2V = None
P2V = None
GRAPH = None
PY2Y = {}

POS_LEN = 43
W2V_LEN = 100
HALF_WINDOW_LEN = 5

#载入w2v
def loadw2v(datapath):
	global W2V
	w2vpath = datapath + "/dim100_word.w2v.bin"
	W2V = gensim.models.KeyedVectors.load_word2vec_format(w2vpath, binary=True, unicode_errors='ignore')
#载入词性的vec
def loadp2v(datapath):
	global P2V
	p2vpath = datapath + "/pos2vec.npy"
	P2V = np.load(p2vpath, allow_pickle=True).item()

def run_frontend(input, output):
	os.system("./frontend_newseg ./../../../tts_svn/tts_frontend_new/data {} > {}".format(input, output))

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
			#print(line)
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

puncSen = ["。", "！", "？"]
punSubSen = ["：", "，", "、"]
punOther = ["…", "“", "”", "《", "》", "（", "）", "～", "·", "‧"]

puncCh = puncSen + punSubSen + punOther

def is_punc(uchar):
	if uchar in string.punctuation or uchar in puncCh:
		return True
	else:
		return False

def regularWord(word):
	patE = re.compile("[a-zA-Z]+")
	patN = re.compile("(-?)([0-9]+)(\.\d+)?")
	patN2 = re.compile("[零一二三四五六七八九十百千万亿]+")

	word = re.sub(patE, "e", word)
	word = re.sub(patN, "n", word)
	word = re.sub(patN2, "n", word)
	newword = ""
	for zi in word:
		#简单化处理，标点符号、tab都处理为p，也可考虑对标点进行分类
		if is_punc(zi) or zi == "	":
			newword += 'p'
		else:
			newword += zi
	return newword


def trans_to_dnn_train(hanzi, seg, label, outputDir):
	if os.path.exists(outputDir + "/" + hanzi + "_x.bin") and os.path.exists(outputDir + "/" + hanzi + "_y.bin"):
		return
	X_featureList = []
	Y_featureList = []
	first_word = W2V[list(W2V.vocab.keys())[0]]
	pad_arr_word = np.ones_like(first_word)
	pad_arr_pos = np.zeros_like(P2V["n"])

	fpseg = codecs.open(seg, "r", "gb18030", errors="ignore")
	fplabel = codecs.open(label, "r", "gb18030", errors="ignore")
	while True:
		#分词结果
		line = fpseg.readline()
		#带拼音
		withlabel = fplabel.readline()

		if (not line.strip() and withlabel.strip()) or (line.strip() and not withlabel.strip()):
			print ("{} and {} is not match!!".format(seg, label))
			exit(1)

		if (not line.strip() and not withlabel.strip()):
			break

		labels = []

		print ("seg result: ", line[:50], "...")
		print ("label: ", withlabel[:50], "...")

		while withlabel.find(hanzi) >= 0:
			index = withlabel.find(hanzi)
			if index < len(withlabel) - 1 and withlabel[index + 1] == "{":
				end = withlabel.find("}", index)
				if (end < 0):
					print ("can not find } in {}".format(withlabel))
					exit(1)
				label = withlabel[index + 2: end]
				Y = PY2Y[label]
				labels.append(Y)
				withlabel = withlabel[:index] + withlabel[end+1:]
			else:
				labels.append(-1)
				withlabel = withlabel[:index] + withlabel[index+1:]

		cursentence = []
		keyindex = []
		hanzicount = 0
		while line.find(' #') >= 0:
			line = line.replace(' #', '#')
		words = line.strip().split(" ")
		print(words)

		for i in range(0, len(words)):
			try:
				word, pos = words[i].split("#")
			except ValueError:
				print("err at：", words[i])
				exit(0)
			curword = {}
			curword["word"] = word
			curword["pos"] = pos
			curword["label"] = -1
			cursentence.append(curword)
			if word.find(hanzi) >= 0:
				if hanzicount >= len(labels):
					print("{} don't match!!, hanzicount is {} and length of labels is {}".format(line.strip(), hanzicount, len(labels)))
					exit(1)
				cursentence[i]["label"] = labels[hanzicount]
				startindex = 0
				count = 0
				while word.find(hanzi, startindex) >= 0:
					count += 1
					k = word.find(hanzi, startindex)
					startindex = k + 1

				for j in range (0, count):
					if labels[hanzicount+j] >= 0:
						keyindex.append(i)
						break
				hanzicount += count

		if hanzicount != len(labels):
			print("{} don't match!!, hanzicount is {} and length of labels is {}".format(line.strip(), hanzicount,
																						len(labels)))
			exit(1)
		#cursentence构建完成, 下一步构建模型输入向量
		if len(keyindex) <= 0:
			print("{} have no key word!!".format(withlabel))
			exit(1)

		for i in keyindex:
			X = np.zeros((HALF_WINDOW_LEN * 2 + 1, W2V_LEN + POS_LEN), dtype=np.float32)
			for j in range(0, HALF_WINDOW_LEN * 2 + 1):
				if i + j - HALF_WINDOW_LEN >= 0 and i + j - HALF_WINDOW_LEN < len(cursentence):
					try:
						w = regularWord(cursentence[i+j-HALF_WINDOW_LEN]["word"])
						w_id = W2V[w]
					except:
						print("can not find {} in w2v".format(cursentence[i+j-HALF_WINDOW_LEN]["word"]))
						w_id = pad_arr_word
					try:
						pos_id = P2V[cursentence[i+j-HALF_WINDOW_LEN]["pos"]]
					except:
						pos_id = pad_arr_pos
					X[j, :W2V_LEN] = w_id
					X[j, W2V_LEN:] = pos_id
			X_featureList.append(X)
			Y_featureList.append(cursentence[i]["label"])
	fpseg.close()
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
				del_dir(c_path)
			else:
				os.remove(c_path)
		os.rmdir(path)



def genData(dataDir):
	# 使用frontend运行train/test文件夹下面所有的withoutlabel文件
	loadp2v("w2v")
	loadw2v("w2v")

	#f = codecs.open("pinyin2label.txt", "r", "gb18030")

	#del_dir("dnn_train_data_word_pos")
	if not os.path.exists("dnn_train_data_word_pos"):
		os.mkdir("dnn_train_data_word_pos")
	if not os.path.exists("word_train_middle"):
		os.mkdir("word_train_middle")

	# 先转换training data
	trainInputList = GetFileList("{}/without_label".format(dataDir), [])

	#hanziFile = codecs.open("temp_list.txt", "r", "utf8", errors="ignore")
	#lines = hanziFile.readlines()

	for inputFile in trainInputList:
	#for line in lines:
		if inputFile.find("_test") >= 0:
			hanzi = inputFile[-10]
		else:
			hanzi = inputFile[-5]
		#hanzi = line[0]
		middleFile = "word_train_middle/" + hanzi + ".txt"
		print(hanzi)
		if not os.path.exists(middleFile):
			run_frontend(inputFile, middleFile)
		if inputFile.find("_test") >= 0:
			labelFile = "{}/with_label/".format(dataDir) + hanzi + "_test.txt"
		else:
			labelFile = "{}/with_label/".format(dataDir) + hanzi + ".txt"
		outDir = "dnn_train_data_word_pos/"
		outFile = outDir + hanzi + "_x.bin.npy"
		if not os.path.exists(outFile):
			#print("genpy2y")
			genPY2Y(labelFile)
			#print("savepy2y")
			savePY2Y(outDir + hanzi + ".dict")
			print (PY2Y)
			#print("trans2dnntrain")
			trans_to_dnn_train(hanzi, middleFile, labelFile, outDir)


genData(sys.argv[1])