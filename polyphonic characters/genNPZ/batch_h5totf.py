import os

def GetFileList(dir, fileList, postfix):
	if os.path.isfile(dir):
		if dir.find("." + postfix) >= 0:
			fileList.append(dir)
	elif os.path.isdir(dir):
		for s in os.listdir(dir):
			newDir = os.path.join(dir, s)
			GetFileList(newDir, fileList, postfix)
	return fileList

#路径参数
input_path = '../model'

h5_list = GetFileList(input_path, [], 'h5')
for weight_file in h5_list:
	os.system("python single_h5totf.py {}".format(weight_file))


