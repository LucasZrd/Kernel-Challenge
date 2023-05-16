import numpy as np
import networkx as nx
from data_parser import get_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def get_laplacian_eigenvalues(g):
    eigenvalues, _ = np.linalg.eig(nx.laplacian_matrix(g).toarray())
    return eigenvalues

training_data, training_labels, test_data = get_data()

g = training_data[np.random.randint(len(training_data))]

def get_features_laplacian_matrix(g):
    eigenvalues = get_laplacian_eigenvalues(g)
    features = []
    features.append(np.sum(eigenvalues < 10**(-10)))
    features.append(np.mean(eigenvalues))
    return features

def number_edges_labels(g):
    labels_edges = []
    for k, v in nx.get_edge_attributes(g,'labels').items():
        labels_edges.append(v[0])
    labels_edges = np.array(labels_edges)
    n_edges_labels = np.zeros(4)
    for i in range(4):
        n_edges_labels[i] = np.sum(labels_edges == i)
    return n_edges_labels/max(1,np.sum(n_edges_labels))

def random_walk(g,k):
    n = len(g.nodes)
    node = np.random.randint(n)
    for i in range(k):
        try:
            node = np.random.choice([n for n in g.neighbors(node)])
        except Exception:
            break
    return node

def last_label_random_walk(g,k,n_iter):
    labels_nodes = nx.get_node_attributes(g,"labels")
    final_labels = np.zeros(50)
    for i in range(n_iter):
        node = random_walk(g,k)
        final_labels[labels_nodes[node][0]] += 1
    return final_labels/np.sum(final_labels)


X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size=0.33)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', class_weight="balanced", probability = True))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X_t = []
for g in X_train:
    features = np.concatenate([number_edges_labels(g),last_label_random_walk(g,2,500)])
    X_t.append(features)

X_v = []
for g in X_test:
    features = np.concatenate([number_edges_labels(g),last_label_random_walk(g,2,500)])
    X_v.append(features)

clf.fit(X_t,y_train)

output = clf.predict_log_proba(X_v)

score = roc_auc_score(y_test, output[:,0])
print(score)
