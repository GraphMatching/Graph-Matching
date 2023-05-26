import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim)
		self.residual_weight = glorot_init(input_dim, output_dim)
		self.activation = activation

	def forward(self, inputs, adj):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(adj, x)
		outputs = self.activation(x)
		return outputs

class GCN(nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim,
				 activation, use_input_augmentation, use_output_augmentation):
		super(GCN, self).__init__()
		self.input_conv = GraphConvSparse(input_dim, hidden_dim, activation)
		if not use_input_augmentation:
			self.hidden_input_dim = hidden_dim
		else:
			self.hidden_input_dim = hidden_dim + input_dim
		self.hidden_convs = nn.ModuleList([GraphConvSparse(self.hidden_input_dim, hidden_dim, activation)
										   for _ in range(num_hidden_layers)])
		self.use_input_augmentation = use_input_augmentation
		self.use_output_augmentation = use_output_augmentation
		if not use_output_augmentation:
			self.output_input_dim = hidden_dim
		else:
			self.output_input_dim = (num_hidden_layers+1)*hidden_dim + input_dim
		self.output_conv = GraphConvSparse(input_dim=self.output_input_dim,
							output_dim=output_dim,
							activation=lambda x:x)

	def forward(self, x, initial_x, adj):
		x_list = [initial_x]
		x = self.input_conv(x, adj)
		x_list.append(x)
		for conv in self.hidden_convs:
			if(self.use_input_augmentation):
				x = conv(torch.cat([initial_x,x],dim=1), adj)
			else:
				x = conv(x, adj)
			x_list.append(x)
		if(self.use_output_augmentation):
			return self.output_conv(torch.cat(x_list,dim=1), adj)
		else:
			return self.output_conv(x, adj)


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

class GINConv(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, A, X):
		#print((X + A @ X).shape)
		X = self.linear(X + A @ X)
		X = torch.nn.functional.relu(X)
		return X


class GIN(torch.nn.Module):
	"""
	implementation adopted from
	https: // github.com / matiasbattocchia / gin
	"""
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
				 use_input_augmentation):
		super().__init__()
		self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
		self.convs = torch.nn.ModuleList()
		self.use_input_agumentation = use_input_augmentation
		if(use_input_augmentation):
			self.hidden_input_dim = input_dim+hidden_dim
		else:
			self.hidden_input_dim = hidden_dim
		for _ in range(n_layers):
			self.convs.append(GINConv(self.hidden_input_dim, hidden_dim))
		self.out_proj = torch.nn.Linear(hidden_dim * (1 + n_layers), output_dim)

	def forward(self, A, X):
		initial_X = torch.empty_like(X).copy_(X)
		X = self.in_proj(X)
		hidden_states = [X]
		for layer in self.convs:
			if(self.use_input_agumentation):
				X = layer(A, torch.cat([initial_X,X],dim=1))
			else:
				X = layer(A, X)
			hidden_states.append(X)
		X = torch.cat(hidden_states, dim=1)
		X = self.out_proj(X)
		return X


class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers, k, no_nodes):
		super().__init__()
		self.k = k
		self.input_size = no_nodes + (k-1)*input_dim
		self.hidden_size = hidden_dim
		self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
		self.hidden_layers = torch.nn.ModuleList()
		self.activation = F.relu
		for _ in range(n_hidden_layers):
			self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
		self.output_layer = torch.nn.Linear(self.hidden_size, output_dim)

	def forward(self, S, X):
		input = self.generate_input(S,X)
		input = self.activation(self.input_layer(input))
		for layer in self.hidden_layers:
			input = self.activation(layer(input))
		return self.activation(self.output_layer(input))

	def generate_input(self, S, X):
		input_list = [S.to_dense()]
		for i in range(1,self.k):
			tmp = S
			for j in range(i):
				tmp = torch.mm(tmp,S.to_dense())
			tmp = torch.mm(tmp,X)
			input_list.append(tmp)
		input = torch.cat(input_list,dim=1)
		return input


def FilterFunction(h, S, x, b=None):
	K = h.shape[1]
	output_dim = h.shape[2]
	y = torch.zeros((S.shape[0], output_dim)).to(S.get_device())
	for k in range(K):
		h_k = h[:,k,:]
		S_k = torch.empty_like(S).copy_(S)
		for i in range(k):
			S_k = torch.matmul(S_k, S.to_dense())
		tmp = torch.matmul(S_k,x)
		tmp = torch.matmul(tmp, h_k)
		y =  torch.add(y, tmp)
	if b is not None:
		y = torch.add(y,b.to(y.get_device()))
	return y


class GraphFilter(nn.Module):
	def __init__(self, k, f_in, f_out,bias):
		super().__init__()
		self.k = k
		self.f_in = f_in
		self.f_out = f_out
		self.weight = nn.Parameter(torch.randn(self.f_in, self.k, self.f_out))
		self.reset_parameters()
		self.bias = bias
	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.f_in * self.k)
		self.stdv = stdv
		self.weight.data.uniform_(-stdv, stdv)


	def forward(self, S, x):
		if self.bias:
			b = nn.parameter.Parameter(torch.Tensor(S.shape[0], self.f_out))
			b.data.uniform_(-self.stdv, self.stdv)
		else:
			b = None
		return FilterFunction(self.weight, S, x, b)


class GNN(nn.Module):
	def __init__(self, num_hidden_layers,input_dim,hidden_dim,
				output_dim,
				activation,
				use_input_augmentation,
				use_output_augmentation,
				 bias):
		super().__init__()
		self.l = num_hidden_layers+2
		self.k = [6,5,4,3,3,3,3,1]
		self.sigma = activation
		self.hidden_layers = torch.nn.ModuleList()
		self.input_layer = GraphFilter(self.k[0], input_dim, hidden_dim,bias)
		self.use_input_augmentation = use_input_augmentation
		self.use_output_augmentation = use_output_augmentation
		self.bias = bias
		for layer in range(num_hidden_layers):
			if(use_input_augmentation):
				self.hidden_layers.append(GraphFilter(self.k[layer+1], input_dim+hidden_dim, hidden_dim,bias))
			else:
				self.hidden_layers.append(GraphFilter(self.k[layer+1], hidden_dim, hidden_dim,bias))
		if not use_output_augmentation:
			self.output_input_dim = hidden_dim
		else:
			self.output_input_dim = (num_hidden_layers+1)*hidden_dim + input_dim

		self.output_layer = GraphFilter(self.k[-1], self.output_input_dim, output_dim,bias)


	def forward(self, S, initial_x):
		x_list = [initial_x]
		x = self.sigma(self.input_layer(S,initial_x))
		x_list.append(x)
		for layer in self.hidden_layers:
			if(self.use_input_augmentation):
				x = self.sigma(layer(S, torch.cat([initial_x,x],dim=1)))
			else:
				x = self.sigma(layer(S, x))
			x_list.append(x)
		if(self.use_output_augmentation):
			return self.output_layer(S, torch.cat(x_list,dim=1))
		else:
			return self.output_layer(S, x)



class GAE(nn.Module):
	"""
	implementation adopted from
	https://github.com/DaehanKim/vgae_pytorch
	"""
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim, activation,
				 use_input_augmentation, use_output_augmentation, encoder):
		super(GAE,self).__init__()
		self.use_input_augmentation =- use_input_augmentation
		self.use_output_augmentation = use_output_augmentation
		self.encoder = encoder
		if(encoder == "GCN"):
			self.base_gcn = GCN(num_hidden_layers=num_hidden_layers,
								input_dim=input_dim,
								hidden_dim=hidden_dim,
								output_dim=output_dim,
								activation=activation,
								use_input_augmentation = use_input_augmentation,
								use_output_augmentation = use_output_augmentation)
		elif(encoder == "GNN"):
			self.base_gcn = GNN(num_hidden_layers=num_hidden_layers,
								input_dim=input_dim,
								hidden_dim=hidden_dim,
								output_dim=output_dim,
								activation = activation,
								use_input_augmentation = use_input_augmentation,
								use_output_augmentation = use_output_augmentation,
								bias = False)
		elif(encoder == "GIN"):
			self.base_gcn = GIN(input_dim, hidden_dim, output_dim, num_hidden_layers+2,
								use_input_augmentation = use_input_augmentation)
		elif(encoder == "MLP"):
			self.base_gcn = MLP(input_dim, hidden_dim, output_dim, num_hidden_layers, 8, 198)
		else:
			print("Encoder model not defined")
			exit()

	def encode(self, X, initial_X, adj):
		if(self.encoder == "GCN"):
			hidden = self.base_gcn(X, initial_X, adj)

		elif(self.encoder == "GIN" ):
			hidden = self.base_gcn(adj, initial_X)

		elif(self.encoder == "MLP"):
			hidden = self.base_gcn(adj, initial_X)

		elif(self.encoder == "GNN"):
			hidden = self.base_gcn(adj, initial_X)

		else:
			print("Not supoorted model")
			exit()

		return hidden

	def forward(self, X, initial_X, adj):
		Z = self.encode(X, initial_X, adj)
		return Z