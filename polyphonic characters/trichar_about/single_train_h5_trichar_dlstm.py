#coding: utf-8
#Simple CNN
import numpy as np
import random
import sys
import os
from keras import Input
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Lambda
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Embedding, LSTM, Bidirectional
#from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import MaxPooling1D

from keras.layers import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from enum import Enum
import datetime
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import codecs
import keras.backend as K
import matplotlib.pyplot as plt

from keras.layers import *


outputDir = "model_dlstm_trichar"

def f1_score(y_true, y_pred):
	# Count positive samples.
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	# If there are no true samples, fix the F1 score at 0.
	if c3 == 0:
		return 0
	# How many selected items are relevant?
	precision = c1/c2
	# How many relevant items are selected?
	recall = c1/c3
	# Calculate f1_score
	f1_score = 2 * (precision * recall)/(precision + recall)
	return f1_score

window = 15
vecdim = 100

seed = 7
np.random.seed(seed)

scoreDict = {}

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

def GetFileList(dir, fileList, postfix):
	if os.path.isfile(dir):
		if dir.find("." + postfix) >= 0:
			fileList.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			GetFileList(newDir, fileList, postfix)
	return fileList
'''
def d_lstm(num_classes):
	model = Sequential()
	model.add(Bidirectional(LSTM(64, return_sequences=False, input_shape=(window, vecdim))))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	model.summary()
	return model
'''
def my_model(num_classes):
	input_tensor = Input(shape=(window, vecdim))
	x = Masking(mask_value=0)(input_tensor)
	x = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(x)
	x = Lambda(lambda x: x[0][:,(window - 1) // 2,:])(x)
	x = Dropout(0.2)(x)
	x = Dense(num_classes, activation='softmax')(x)
	model = Model(input_tensor, x)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	model.summary()
	return model



def data_shuffle(x, y):
	z = list(zip(x, y))
	random.shuffle(z)
	a, b = zip(*z)
	return np.array(a), np.array(b)

def drawHistroy(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='validation acc')
	plt.title("training and validation accuracy")

	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='validation loss')
	plt.title("training and validation loss")
	plt.legend()
	plt.show()

x_total = np.array([])
y_total = np.array([])
num_classes = 0

def load_data(hanzi):
	global x_total
	global y_total
	global num_classes
	x_total = np.load("dnn_train_data_trichar/{}_x.bin.npy".format(hanzi))
	y_total = np.load("dnn_train_data_trichar/{}_y.bin.npy".format(hanzi))

	#x_total = x_total.reshape(x_total.shape[0], window, vecdim, 1).astype('float32')
	x_total = x_total.reshape(x_total.shape[0], window, vecdim).astype('float32')
	y_total = np_utils.to_categorical(y_total)
	# 乱序
	x_total, y_total = data_shuffle(x_total, y_total)
	num_classes = y_total.shape[1]

def try_method(params, resultFile):
	global x_total
	global y_total
	global num_classes
	print("trainning params: {}".format(params))
	# k 折交叉验证
	k = 4
	num_val_samples = len(x_total) // k
	all_score = []
	for i in range(k):
		val_x = x_total[i*num_val_samples : (i+1)*num_val_samples]
		val_y = y_total[i*num_val_samples : (i+1)*num_val_samples]

		train_x = np.concatenate([x_total[:i*num_val_samples], x_total[(i+1)*num_val_samples:]], axis=0)
		train_y = np.concatenate([y_total[:i*num_val_samples], y_total[(i+1)*num_val_samples:]], axis=0)

		model = cnn_2d_dense_model(params[:-2], num_classes)

		history = model.fit(train_x, train_y, epochs=params[-2], batch_size=params[-1], class_weight='auto',
						verbose=0)

		val_loss, val_acc = model.evaluate(val_x, val_y, verbose=0)
		#print("val loss: {} val acc: {}".format(val_loss, val_acc))
		#drawHistroy(history)
		#model.save("model/{}.h5".format(hanzi))
		all_score.append(val_acc)
		#model.summary()

	ret = np.mean(all_score)
	print("all scores: {}, mean: {}".format(all_score, ret))
	resultFile.write("params: {}	mean score: {}\n".format(params, ret))
	return ret

def train_save(hanzi, params, resultFile):
	global x_total
	global y_total
	global num_classes

	#model = d_lstm(num_classes)
	model = my_model(num_classes)
	print(x_total.shape)

	# Fit the model
	history = model.fit(x_total, y_total, validation_split=0.2, epochs=params[-2], batch_size=params[-1], class_weight='auto', verbose=2)
	model.save("{}/{}.h5".format(outputDir, hanzi))

	val_acc = history.history['val_acc']
	resultFile.write("params: {}\n".format(params))
	resultFile.write("val_acc: {}\n".format(val_acc))
	resultFile.write("\n")
	return val_acc
'''
def try_params(hanzi, params):
	if not os.path.exists(outputDir):
		os.mkdir(outputDir)
	resultFile = codecs.open("{}/evaluate_{}_{}.txt".format(outputDir, params, hanzi), 'w', 'gb18030')

	bestResult = 0
	bestParams = []
	load_data(hanzi)

	epochsList = [10, 31, 5]
	batchSizeList = [16, 32, 64]
	for ep in epochsList:
		for bs in batchSizeList:
			params_t = params.copy()
			params_t.append(ep)
			params_t.append(bs)
			result = try_method(params_t, resultFile)
			if result > bestResult:
				bestResult = result
				bestParams = params_t

	print("best result:{}	best params:{}".format(bestResult, bestParams))
	resultFile.close()
'''
def use_params(hanzi):
	if not os.path.exists(outputDir):
		os.mkdir(outputDir)
	params = [50, 32]
	resultFile = codecs.open("{}/evaluate_{}_{}.txt".format(outputDir, params, hanzi), 'w', 'gb18030')

	load_data(hanzi)

	val_acc = train_save(hanzi, params, resultFile)
	resultFile.close()
	return val_acc

def do_train(zi):
	print("train {}:".format(zi))
	return use_params(zi)

if __name__=='__main__':
	hanzi = sys.argv[1]
	logFile = sys.argv[2]
	val_acc = do_train(hanzi)
	fp = codecs.open(logFile, "a", "gb18030")
	fp.write("{}".format(hanzi))
	for val in val_acc:
		fp.write("	%.4f" % val)
	fp.write("\n")
	fp.close()
