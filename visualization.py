import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from data_parser import get_data

training_data, training_labels, test_data = get_data()

def plot_graph(g):
    labels_edges = []
    for k, v in nx.get_edge_attributes(g,'labels').items():
        labels_edges.append(v[0])

    clean_labels_nodes = {}
    labels_nodes = nx.get_node_attributes(g,"labels")
    for k, v in labels_nodes.items():
        clean_labels_nodes[k] = v[0]

    nx.draw(g, edge_color = labels_edges, labels = clean_labels_nodes, with_labels = True)
    plt.show()

u = np.random.randint(len(training_data))

g = training_data[u]
label = training_labels[u]

print(label)
plot_graph(g)




