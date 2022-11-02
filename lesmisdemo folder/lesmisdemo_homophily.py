import numpy as np
import networkx as nx
import random
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


#parameters
file = "lesmis.edgelist"
output_file = "lesmis.emb"
weighted = True
directed = False
dimensions = 20
workers = 8 #cpu cores
epochs = 1
window_size = 6
p = 0.5
q = 3
num_walks = 100
walk_length = 20

n_cluster = 6

plot = True

def main():
	#load our graph
	G = node2vec(file, weighted,directed, p, q)
	
	#simulate walks and convert it into a list
	walks = G.simulate_walks(num_walks, walk_length)
	walks = [list(map(str, walk)) for walk in walks]

	#model the walks using Word2Vec package
	model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, epochs=epochs)
	model.wv.save_word2vec_format(output_file)   
    

	# Kmeans
	vectors_lst =[model.wv[str(n)] for n in G.ntw.nodes()]    # list of vectors(for each node), order like in G.ntw.nodes()       
	   
	kmeans = KMeans(n_clusters=n_cluster).fit(vectors_lst)
	#print(type(kmeans.labels_))                         
	
	# Draw the graph
	color_list = labels_to_colors(kmeans.labels_)  # converting the array with labels(cluster numbers) for each node to list of colors, order like in G.ntw.nodes()
	degrees = dict(G.ntw.degree)
	
	nx.draw(G.ntw, with_labels = False, node_size=[v * 75 for v in degrees.values()], alpha = 0.8, node_color = color_list)
	plt.show()

def labels_to_colors(x):
	# List of labels(numbers of clusters) to list of colors
	colors_dict = {}
	colors_dict[0] = 'maroon'
	colors_dict[1] = 'paleturquoise'
	colors_dict[2] = 'limegreen'
	colors_dict[3] = 'orange'
	colors_dict[4] = 'gray'
	colors_dict[5] = 'teal'
	
	color_list = [colors_dict[l] for l in x]   # order like in G.ntw.nodes()
	return color_list

class node2vec():
	def __init__(self, file,weighted, directed, p, q):
		#We load the graph
		if weighted:
			ntw = nx.read_edgelist(file, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
		else:
			ntw = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph())
			for edge in ntw.edges():
				ntw[edge[0]][edge[1]]['weight'] = 1

		if not directed:
			ntw = ntw.to_undirected()
		
		self.directed = directed
		self.p = p
		self.q = q

		self.ntw = ntw

		alias_nodes = {}
		for node in ntw.nodes():
			unnormalized_probs = [self.ntw[node][nbr]['weight'] for nbr in sorted(self.ntw.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}

		if directed:
			for edge in ntw.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in ntw.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges



	def node_walk(self, walk_length, start_node):
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		for _ in range(walk_length):
			cur = walk[-1]
			cur_nbrs = sorted(self.ntw.neighbors(cur))
			#if len(cur_nbrs) > 0:
			if len(walk) == 1:
				walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
			else:
				prev = walk[-2]
				next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
					alias_edges[(prev, cur)][1])]
				walk.append(next)
			#else:
			#	break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		walks = []
		nodes = list(self.ntw.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(f"{walk_iter+1}/{num_walks}")
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(self.ntw.neighbors(dst)):
			unnormalized_probs.append(self.ntw[dst][dst_nbr]['weight'])
			if dst_nbr == src:
				unnormalized_probs[-1] /= p
			elif not self.ntw.has_edge(dst_nbr, src):
				unnormalized_probs[-1] /= q
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)



def alias_setup(probs):
	
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]

if __name__ == "__main__":
    main()