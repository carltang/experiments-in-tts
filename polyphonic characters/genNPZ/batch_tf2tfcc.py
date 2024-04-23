#import argparse
#import tensorflow as tf
#import numpy as np
#import struct
import codecs
import os
import sys

def GetFileList(dir, fileList, postfix):
	if os.path.isfile(dir):
		if dir.find("." + postfix) >= 0:
			fileList.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			GetFileList(newDir, fileList, postfix)
	return fileList

pb_list = GetFileList("pb", [], 'pb')

#recodeFile = codecs.open("polySet.txt", "w", "gb18030")

if not os.path.exists("tfcc"):
      os.makedirs("tfcc")

#num = 0
for pbfile in pb_list:
    hanzi = pbfile[-4]
    npzfile = "tfcc/" + hanzi + ".npz"
    os.system("python single_tf2tfcc.py --checkpoint {} --out {}".format(pbfile, npzfile))
    #recodeFile.write("{}\n".format(hanzi))
    #print(hanzi)
    #os.system("cp ../dnn_train_data_word_pos/{}.dict tfcc/{}.dict".format(hanzi, num))
    #num += 1

#recodeFile.close()