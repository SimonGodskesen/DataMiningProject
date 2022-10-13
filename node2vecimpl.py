from curses import raw
from operator import truediv
import numpy as np
import networkx as nx
import random
import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from vose_sampler import VoseAlias



#parameters
file = "karate.edgelist"
output_file = "karate.csv"
weighted = False
directed = False
dimensions = 128
workers = 8 #cpu cores
epochs = 10
window_size = 1
p = 1
q = 5
num_walks = 10
walk_length = 80

n_cluster = 3

plot = False

def main():
	#We load the graph
	#redefine it into our own graph-system
	G = node2vec(file,weighted, directed, p, q)
	
	#simulate walks and convert it into a list
	walks = G.walks(num_walks, walk_length)
	walks = [list(map(str, walk)) for walk in walks]

	#model the walks using Word2Vec package
	model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, epochs=epochs)
	model.wv.save_word2vec_format(output_file)   

	#Cluster it using K-means
	vectored_nodes = pd.read_csv(output_file, sep=' ', skiprows=[0], header=None)
	kmeans = KMeans(n_clusters=n_cluster)
	kmeans.fit(vectored_nodes)
	labels = kmeans.predict(vectored_nodes)
	label_dict = {}
	for i,e in enumerate(labels):
		label_dict[i+1] = e
	
	#plotting
	if plot:
		nx.draw(G,labels=label_dict,node_color = 'black',font_color = "yellow")
		plt.savefig("figure")
		plt.show()
	

class node2vec():
    def __init__(self,file,weighted,directed,p,q):
        self.weighted = weighted
        self.directed = directed
        self.p = p
        self.q = q

        if weighted:
            G = nx.read_edgelist(file, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1
        
        node_alias = {}
        for node in G.nodes():
            unnormalized = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            normalized = [float(e)/sum(unnormalized) for e in unnormalized]
            #print(G.neighbors(node))
            node_alias[node] = self.alias_setup(normalized)

        edge_alias = {}

        def _get_edge(start,dest):
            un = []
            for nbr in sorted(G.neighbors(dest)): 
                un.append(G[dest][nbr]['weight'])             
                if nbr == start:
                    un[-1] /= p
                elif not G.has_edge(dest,start):
                    un[-1] /= q
            n = [float(e)/sum(un) for e in un]
            return self.alias_setup(n)

        for edge in G.edges():
            edge_alias[edge] = _get_edge(edge[0],edge[1])
            if not directed:
                edge_alias[(edge[1],edge[0])] = _get_edge(edge[1],edge[0])
           
        self.G = G
        self.edge_alias = edge_alias
        self.node_alias = node_alias
    
    def walks(self,n,m):
        walks = []
        nodes = list(self.G.nodes())
        for _ in range(n):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node_walk(m=m,node=node))
        return walks
    
    def node_walk(self,m,node):
        walk = [node]

        for _ in range(m):
            here = walk[-1]
            print("here",here)
            nbrs = sorted(self.G.neighbors(here))
            if len(nbrs) > 0:
                if len(walk) == 1:
                    print("ny print ",self.node_alias[node])
                    walk.append(self.alias_draw(self.node_alias[here][0],self.node_alias[here][1]))
                else:
                    there = walk[-2]
                    print("edge:", self.edge_alias[(there,here)])
                    print("next node index",self.alias_draw(self.edge_alias[(there,here)][0],self.edge_alias[(there,here)][1]))
                    next_node = nbrs[self.alias_draw(self.edge_alias[(there,here)][0],self.edge_alias[(there,here)][1])]
                    walk.append(next_node)
            else:
                break
        return walk
    

    def alias_setup(self,probs):
        
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


    def alias_draw(self,J, q):
        K = len(J)

        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]



if __name__ == "__main__":
    main()
        


