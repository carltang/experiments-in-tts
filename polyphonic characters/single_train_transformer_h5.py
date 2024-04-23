#coding: utf-8
#Simple CNN
import numpy as np
import random
import sys
import os
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers.recurrent import LSTM
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


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output



'''
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

window = 11
vecdim = 143

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

def cnn_2d_dense_model(params, num_classes):
	# create model
	model = Sequential()
	model.add(Conv2D(params[0], (params[1], params[2]), strides=(1, params[3]), input_shape=(window, vecdim, 1), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Conv2D(16, (3, 5), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(params[8]))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(params[4], activation='relu'))
	#model.add(Dense(1, activation='sigmoid'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_score])
	return model

def cnn_2d_dense_model_old(params, num_classes):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (2, 4), input_shape=(11, 143, 1), activation='relu'))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_score])
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
	x_total = np.load("dnn_train_data_word_pos/{}_x.bin.npy".format(hanzi))
	y_total = np.load("dnn_train_data_word_pos/{}_y.bin.npy".format(hanzi))
	# print("load trainning data shape: ", x_total.shape)
	x_total = x_total.reshape(x_total.shape[0], window, vecdim, 1).astype('float32')
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

	#x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=1)

	# build the model
	model = cnn_2d_dense_model(params[:-2], num_classes)

	# Fit the model
	history = model.fit(x_total, y_total, validation_split=0.2, epochs=params[-2], batch_size=params[-1], class_weight='auto', verbose=2)
	model.save("model/{}.h5".format(hanzi))

	val_acc = history.history['val_acc']

	resultFile.write("params: {}\n".format(params))
	resultFile.write("val_acc: {}\n".format(val_acc))
	resultFile.write("\n")
	return val_acc

def try_params(hanzi, params):
	if not os.path.exists("model"):
		os.system("mkdir model")
	resultFile = codecs.open("model/evaluate_{}_{}.txt".format(params, hanzi), 'w', 'gb18030')

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

def use_params(hanzi, params):
	if not os.path.exists("model"):
		os.system("mkdir model")
	resultFile = codecs.open("model/evaluate_{}_{}.txt".format(params, hanzi), 'w', 'gb18030')

	load_data(hanzi)

	params_t = params.copy()
	params_t.append(50)
	params_t.append(32)
	val_acc = train_save(hanzi, params_t, resultFile)
	resultFile.close()
	return val_acc

def do_train(zi):
	print("train {}:".format(zi))
	params = [32, 2, 16, 15, 32]
	#try_params(params)
	return use_params(zi, params)

if __name__=='__main__':
	hanzi = sys.argv[1]
	val_acc = do_train(hanzi)
	fp = codecs.open("train_results.txt", "a", "gb18030")
	fp.write("{}".format(hanzi))
	for val in val_acc:
		fp.write("	%.4f" % val)
	fp.write("\n")
	fp.close()
'''
