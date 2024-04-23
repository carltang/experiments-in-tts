	 # coding=utf-8
from __future__ import print_function
import codecs
import re
import os
import sys

def getFileList(dir, fileList):
	if os.path.isfile(dir):
		if dir.find(".txt") >= 0:
			fileList.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			getFileList(newDir, fileList)
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

def getZiSet(fileList):
	ziSet = set()
	for l in fileList:
		if l.find("_test") >= 0:
			zi = l[-10]
		else:
			zi = l[-5]
		ziSet.add(zi)
	return ziSet

def getLineList(fileName):
	if os.path.exists(fileName):
		fp = codecs.open(fileName, 'r', "gb18030")
		lines = fp.readlines()
		return lines
	else:
		return []

def writeLines(fp, lines):
	for l in lines:
		fp.write(l)


del_dir("combin_data")
os.mkdir("combin_data")
dir_l = "combin_data/with_label"
os.mkdir(dir_l)
dir_n = "combin_data/without_label"
os.mkdir(dir_n)

def combin():
	dir1 = "combin_data_y1+b1"
	dir2 = "data_yunhui_3/"
	filelist1 = getFileList(os.path.join(dir1, "with_label"), [])
	filelist2 = getFileList(os.path.join(dir2, "with_label"), [])
	ziSet1 = getZiSet(filelist1)
	ziSet2 = getZiSet(filelist2)
	combinSet = ziSet1 | ziSet2

	for zi in combinSet:
		print (zi)
		path1a = os.path.join(dir1, "with_label", zi + ".txt")
		path1b = os.path.join(dir1, "without_label", zi + ".txt")
		path2a = os.path.join(dir2, "with_label", zi + "_test.txt")
		path2b = os.path.join(dir2, "without_label", zi + "_test.txt")
		path3a = os.path.join(dir_l, zi + ".txt")
		path3b = os.path.join(dir_n, zi + ".txt")

		line1a = getLineList(path1a)
		line2a = getLineList(path2a)
		print(len(line1a), len(line2a))
		fp3a = codecs.open(path3a, 'w', 'gb18030')
		writeLines(fp3a, line1a)
		writeLines(fp3a, line2a)
		fp3a.close()

		line1b = getLineList(path1b)
		line2b = getLineList(path2b)
		fp3b = codecs.open(path3b, 'w', 'gb18030')
		writeLines(fp3b, line1b)
		writeLines(fp3b, line2b)
		fp3b.close()



combin()