import torch
import torch.nn as nn
from torch.nn import init

class Edge_updater_nn(nn.Module):
	def __init__(self, node_input_size, output_size , act = 'tanh',relation_input_size = None, bias = True):
		super(Edge_updater_nn,self).__init__()
		self.h2o = nn.Linear(node_input_size,output_size,bias)
		self.l2o = nn.Linear(node_input_size,output_size,bias)
		if relation_input_size is not None:
			self.r2o = nn.Linear(relation_input_size,output_size,bias)
		if act == 'tanh':
			self.act = nn.Tanh()
		elif act == 'sigmoid':
			self.act = nn.Sigmoid()
		else:
			self.act = nn.ReLU() 


	def forward(self, head_node, tail_node, relation=None):

		if relation is None:
			edge_output = self.h2o(head_node) + self.l2o(tail_node)
		else:
			edge_output = self.h2o(head_node) + self.l2o(tail_node) + self.r2o(relation)
		edge_output_act = self.act(edge_output)
		return edge_output_act

class Combiner(nn.Module):
	def __init__(self, input_size, output_size,act, bias = True ):
		super(Combiner,self).__init__()
		self.h2o = nn.Linear(input_size,output_size,bias)
		self.l2o = nn.Linear(input_size,output_size,bias)
		if act == 'tanh':
			self.act = nn.Tanh()
		elif act == 'sigmoid':
			self.act = nn.Sigmoid()
		else:
			self.act = nn.ReLU() 

	def forward(self, head_info, tail_info):
		node_output = self.h2o(head_info) + self.l2o(tail_info)
		node_output_tanh = self.act(node_output)
		return node_output_tanh
