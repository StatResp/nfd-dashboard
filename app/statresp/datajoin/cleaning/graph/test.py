# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:10:54 2021

@author: vaziris
"""
import networkx as nx
import matplotlib.pyplot as plt
import nxviz as nv
G=nx.Graph()
G.add_node(1)
G.nodes[1]['N']=10
G.add_nodes_from([(2,{'N':20}),3,4])
G.add_edge(1,2)
G.edges[1,2]['E']=120
G.add_edges_from([(1,3,{'E':130}),(1,4)])
G.nodes()
pos = {0: (0, 0),
       1: (1, 0),
       2: (0, 1),
       3: (1, 1),
       4: (0.5, 2.0)}
print(G.nodes(data=True))
print(G.edges(data=True))
nx.draw(G, pos, with_labels=True, font_weight='bold')
FIG=nv.CircosPlot(G,node_size=1)
FIG.draw();plt.show()

Dist_Mat=nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(G,[1, 2,3,4])
print(Dist_Mat)
Dist_Mat=nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(G,[2,3,4])
print(Dist_Mat)



#length = dict(nx.all_pairs_shortest_path_length(G))
length = dict(nx.single_source_shortest_path_length(G,2,100));length
length = dict(nx.multi_source_dijkstra_path_length(G,{1,2}));length
length = dict(nx.single_source_bellman_ford_path_length(G,2,100));length


 

