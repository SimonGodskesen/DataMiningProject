from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
raw_G = nx.read_edgelist("facebook_combined.txt", nodetype=int, create_using=nx.DiGraph())
plot = True
vectored_nodes = pd.read_csv("facebook_result", sep=' ', skiprows=[0], header=None)
kmeans = KMeans(n_clusters=4)
kmeans.fit(vectored_nodes)
labels = kmeans.predict(vectored_nodes)
label_dict = {}
for i,e in enumerate(labels):
    label_dict[i+1] = e


#plotting
if plot:
    nx.draw(raw_G,labels=label_dict,node_color = 'black',font_color = "yellow")
    plt.savefig("figure")
    plt.show()