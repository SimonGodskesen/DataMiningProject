from curses import raw
from operator import truediv
import numpy as np
import networkx as nx
import random
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


#parameters
file = "karate.edgelist"
output_file = "karate.emb"
weighted = False
directed = False
dimensions = 128
workers = 8 #cpu cores
epochs = 10
window_size = 1
p = 1
q = 0.1
num_walks = 10
walk_length = 80

n_cluster = 3

plot = True

def main():
	#load our graph
	G = node2vec(file, weighted,directed, p, q)
	
	#simulate walks and convert it into a list
	walks = G.simulate_walks(num_walks, walk_length)
	walks = [list(map(str, walk)) for walk in walks]    

	#model the walks using Word2Vec package
	model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, epochs=epochs)
	model.wv.save_word2vec_format(output_file)      # save result at "karate.emb" file

	#Cluster it using K-means
	vectored_nodes = pd.read_csv(output_file, sep=' ', skiprows=[0], header=None)
	kmeans = KMeans(n_clusters=n_cluster)
	kmeans.fit(vectored_nodes)
	labels = kmeans.predict(vectored_nodes)
	label_dict = {}
	for i,e in enumerate(labels):
		label_dict[i+1] = e                # dictionary with counts and values of labels
	
	#plotting
	if plot:
		nx.draw(G.ntw,labels=label_dict,node_color = 'black',font_color = "yellow")
		plt.savefig("figure")              # save figure of the graph with clustering to figure.png
		plt.show()
	
#########################################################################################################

class node2vec():
	def __init__(self, file,weighted, directed, p, q):
		#We load the graph
		if weighted:
			ntw = nx.read_edgelist(file, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
		else:
			ntw = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph())
			for edge in ntw.edges():
				ntw[edge[0]][edge[1]]['weight'] = 1   # set weight for each edge = 1 if the graph is not weighted

		if not directed:
			ntw = ntw.to_undirected()
		
		self.directed = directed    # False here
		self.p = p
		self.q = q

		self.ntw = ntw

		alias_nodes = {}
		for node in ntw.nodes():
			unnormalized_probs = [self.ntw[node][nbr]['weight'] for nbr in sorted(self.ntw.neighbors(node))]   # assign weight values to unnormalized probs
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs] # normalized probabilities(based on weights) for all nodes
			alias_nodes[node] = alias_setup(normalized_probs)     # Initialization of alias sampling for nodes from nodes-neighbours, array with probabilities and pointers to alies

		alias_edges = {}

		if directed:
			for edge in ntw.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])  # initialization of alias sampling for every edge(for sampling neighbours of edge[1]), returnes q and J arrays 
		else:
			for edge in ntw.edges():                      # if indirected, then the same in both directions
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes  # alias sampling initialization results: q and J for every node(takes weights in account) 
		self.alias_edges = alias_edges  # alias sampling intialization results: q and J for every node(takes weights, p, q in account) 



	def node_walk(self, walk_length, start_node):
		alias_nodes = self.alias_nodes  # alias sampling initialization results: (based on weights), arrays q and J for all nodes
		alias_edges = self.alias_edges  # alias sampling initialization results: (based on weights, p, q), arrays q and J for all nodes

		walk = [start_node]               # begins the walk from the start node

		for _ in range(walk_length):
			cur = walk[-1]                              # gets the current node
			cur_nbrs = sorted(self.ntw.neighbors(cur))  # gets neighbours from the current node
			#if len(cur_nbrs) > 0:
			if len(walk) == 1:                          # if it is first node of the walk, sample based on weights
				walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]) # sample the next node based on arrays, obtained during initialization of alias sampling(q and J)
			else:
				prev = walk[-2]                         # else previous node can be used, i.e. sampling based on weights, p,q
				next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],        # sample the next node based on arrays, obtained during initialization of alias sampling(q and J)
					alias_edges[(prev, cur)][1])]
				walk.append(next)
			#else:
			#	break

		return walk                                   # return the walk of length walk_length

	def simulate_walks(self, num_walks, walk_length):
		walks = []
		nodes = list(self.ntw.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):      # generate walks num_walks times 
			print(f"{walk_iter+1}/{num_walks}")
			random.shuffle(nodes)
			for node in nodes:                 # generate random walks starting from each node 
				walks.append(self.node_walk(walk_length=walk_length, start_node=node))

		return walks   # list with walks 

	def get_alias_edge(self, src, dst):         # Initialization of alias sampling 
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(self.ntw.neighbors(dst)):    # for every neighbour of destination node
			unnormalized_probs.append(self.ntw[dst][dst_nbr]['weight'])  # weight of edge leading to neighbour
			if dst_nbr == src:
				unnormalized_probs[-1] /= p                           # if returned to the source node, /p
			elif not self.ntw.has_edge(dst_nbr, src):
				unnormalized_probs[-1] /= q                           # if the neighbour of destination node doesn't have an edge wth the source node, /q
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]   #normalization

		return alias_setup(normalized_probs)      # initialization of alias sampling, returns: 
                                                  # q - array with probabilities (of nodes, based on weights, p, q)
                                                  # J - array which points to the allies of the nodes



def alias_setup(probs): # Initialization of alias sampling
	
	K = len(probs)                      # number of neighbour nodes
	q = np.zeros(K)                     # numpy array for Probabilities
	J = np.zeros(K, dtype=np.int)       # numpy array for pointers to Alias for every node

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):           # kk-number, prob - value from normalized probabilities
		q[kk] = K*prob                          # number of nodes * prob (prob is based on weight or weight and p,q of edge leading to this node)
		if q[kk] < 1.0:
			smaller.append(kk)                  # numbers of nodes with K*prob less then 1
		else:
			larger.append(kk)                   # numbers of nodes with K*prob more then 1

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large                       # Set pointer to alias node for a node with p<1(to fill the box) 
		q[large] = q[large] + q[small] - 1.0   # reduce probability in the column of alias node- this part of prob in the column of the node with p<1
		if q[large] < 1.0:
			smaller.append(large)              # if new value of prob in the column of the alias node<1, add to a list smaller
		else:
			larger.append(large)               # # if new value of prob in the column of the alias node>=1, add to a list larger

	return J, q                                # q- array with probabilities
                                               # J - array which points to the allies of the nodes 
                                               # if we'll imagine a rectangle divided on two rectangles- one for the prob. of node(p), one - for the part(1-p) of prob of its allie

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))        # chose randomly the node's number
	if np.random.rand() < q[kk]:                  # if generated rand.number< prob., then take this node
		return kk
	else:
		return J[kk]                              # else take alias of this node

if __name__ == "__main__":
    main()