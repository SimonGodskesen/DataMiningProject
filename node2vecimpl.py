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
            unnormalized = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
            normalized = [float(e)/sum(unnormalized) for e in unnormalized]
            #print(G.neighbors(node))
            node_alias[node] = self.alias_setup(G.neighbors(node),normalized)

        edge_alias = {}

        def _get_edge(start,dest):
            un = []
            lst = []
            for nbr in G.neighbors(dest):   
                lst.append((dest,nbr))
                un.append(G[dest][nbr]['weight'])             
                if nbr == start:
                    un[-1] /= p
                elif not G.has_edge(dest,start):
                    un[-1] /= q
            n = [float(e)/sum(un) for e in un]
            return self.alias_setup(lst,n)

        for edge in G.edges():
            edge_alias[edge] = _get_edge(edge[0],edge[1])
            if not directed:
                edge_alias[(edge[1],edge[0])] = _get_edge(edge[1],edge[0])
           
        self.G = G
        self.edge_alias = edge_alias
        self.node_alias = node_alias

    def alias_setup(self,nodes,probs):
        lex = {}
        for i,e in enumerate(nodes):
            lex[e] = probs[i]
            print(e)
        return VoseAlias(lex)

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
            if len(walk) == 1:
                walk.append(self.node_alias[node].alias_generation())
            else:
                there = walk[-2]
                next_node = self.edge_alias[(there,here)]
                print(next_node.alias_generation())
                walk.append(next_node)
        return walk



if __name__ == "__main__":
    main()
        


