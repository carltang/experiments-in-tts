from keras.models import load_model
import tensorflow as tf
import os
import sys
import os.path as osp
from keras import backend as K
from tensorflow.python.framework import graph_util,graph_io


'''
#转换函数
def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
	if osp.exists(output_dir) == False:
		os.mkdir(output_dir)
	out_nodes = []
	for i in range(len(h5_model.outputs)):
		out_nodes.append(out_prefix + str(i + 1))
		tf.identity(h5_model.output[i],out_prefix + str(i + 1))
	sess = K.get_session()
	init_graph = sess.graph.as_graph_def()
	main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
	graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
	graph_io.write_graph(main_graph, output_dir, name=model_name + ".txt", as_text=True)
	
h5_list = GetFileList(input_path, [], 'h5')

for weight_file in h5_list:
	h5_model = load_model(weight_file)
	output_graph_name = weight_file[-4:-3] + '.pb'
	h5_to_pb(h5_model,output_dir = output_path,model_name = output_graph_name)
print('model saved')
'''

def freeze_session(output_graph_name, keep_var_names=None, output_names=None, clear_devices=True):
	"""
	Freezes the state of a session into a pruned computation graph.

	Creates a new computation graph where variable nodes are replaced by
	constants taking their current value in the session. The new graph will be
	pruned so subgraphs that are not necessary to compute the requested
	outputs are removed.
	@param session The TensorFlow session to be frozen.
	@param keep_var_names A list of variable names that should not be frozen,
						  or None to freeze all the variables in the graph.
	@param output_names Names of the relevant graph outputs.
	@param clear_devices Remove the device directives from the graph for better portability.
	@return The frozen graph definition.
	"""
	session = K.get_session()
	graph = session.graph
	with graph.as_default():
		freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
		output_names = output_names or []
		output_names += [v.op.name for v in tf.global_variables()]
		input_graph_def = graph.as_graph_def()
		if clear_devices:
			for node in input_graph_def.node:
				node.device = ""
		frozen_graph = tf.graph_util.convert_variables_to_constants(
			session, input_graph_def, output_names, freeze_var_names)
		return frozen_graph


#h5_list = GetFileList(input_path, [], 'h5')

weight_file = sys.argv[1]
h5_model = load_model(weight_file)
print (weight_file)
output_graph_name = weight_file[-4:-3] + '.pb'
frozen_graph = freeze_session(output_graph_name, output_names=[out.op.name for out in h5_model.outputs])
tf.train.write_graph(frozen_graph, "pb", output_graph_name, as_text=False)
print('model saved')

