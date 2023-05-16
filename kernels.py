import numpy as np
import networkx as nx
from typing import Union, List, Tuple

#### Classical kernels

class DiracKernel:
    """
    K(x,y) = 1_{x = y}
    """
    def __init__(self):
        pass
    
    def kernel(self, x: np.array, y: np.array) -> float:
        return int(x==y)

class RBF:
    """
    K(x,y) = exp(-(x-y)^2/(2 sigma^2))
    """
    def __init__(self, sigma: float = 1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,x: float, y: float) -> float:
        return  np.exp(-np.linalg.norm(x-y)**2/(2*self.sigma**2))
    
class LinearKernel:
    """
    K(x,y) = x^T y
    """
    def __init__(self):
        pass

    def kernel(self,x: np.array, y: np.array) -> float:
        return np.dot(x,y)

class PolynomialKernel:
    """
    K(x,y) = (x^Ty + c)^d
    """
    def __init__(self, c: float, d: int):
        self.c = c
        self.d = d

    def kernel(self, x: np.array, y: np.array) -> float:
        return (np.dot(x,y) + self.c)**self.d
    
#### Graphs kernels

class AllNodePairsKernel:
    """
    The kernel applying a kernel on each pair of nodes (x,y) in G1xG2
    """
    def __init__(self, node_kernel: Union[PolynomialKernel, DiracKernel, LinearKernel, RBF], normalize: bool = True):
        """
        Args:
            node_kernel (Union[PolynomialKernel, DiracKernel, LinearKernel, RBF]): kernel applied on nodes
            normalize (bool, optional): if True divide the kernel by the total number of nodes in G1 and G2. Defaults to True.
        """
        self.node_kernel = node_kernel
        self.normalize = normalize

    def kernel(self, G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph) -> float:
        V1 = np.array(list(nx.get_node_attributes(G1,"labels").values()))
        V2 = np.array(list(nx.get_node_attributes(G2,"labels").values()))
        if self.normalize:
            return np.sum([np.sum([self.node_kernel(v1,v2) for v1 in V1]) for v2 in V2])/(len(V1) + len(V2))
        return np.sum([np.sum([self.node_kernel(v1,v2) for v1 in V1]) for v2 in V2])
    
class AllEdgePairsKernel:
    """
    The kernel applying a kernel on each pair of edges (x,y) in G1xG2
    """
    def __init__(self, edge_kernel: Union[PolynomialKernel, DiracKernel, LinearKernel, RBF], normalize: bool = True):
        """
        Args:
            node_kernel (Union[PolynomialKernel, DiracKernel, LinearKernel, RBF]): kernel applied on nodes
            normalize (bool, optional): if True divide the kernel by the total number of nodes in G1 and G2. Defaults to True.
        """
        self.edge_kernel = edge_kernel
        self.normalize = normalize

    def kernel(self, G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph) -> float:
        V1 = np.array(list(nx.get_edge_attributes(G1,"labels").values())).flatten()
        V2 = np.array(list(nx.get_edge_attributes(G2,"labels").values())).flatten()
        if self.normalize:
            return np.sum([np.sum([self.node_kernel(v1,v2) for v1 in V1]) for v2 in V2])/(len(V1) + len(V2))
        return np.sum([np.sum([self.node_kernel(v1,v2) for v1 in V1]) for v2 in V2])
    
def direct_product_graph(G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
    """
    Computes the direct product graph of G1 and G2, i.e the graph with nodes (x,y) where x is a node of G1
    and y a node of G2 and label(x) = label(y) and edges ((a,b), (c,d)) if (a,c) edge of G1 and (b,d) edge of G2
    Args:
        G1 (nx.classes.graph.Graph): Graph 1
        G2 (nx.classes.graph.Graph): Graph 2

    Returns:
        Direct product graph Gx = G1 x G2
    """
    Gx = nx.Graph()
    V1 = G1.nodes
    V2 = G2.nodes
    for i in range(len(V1)):
        for j in range(len(V2)):
            if V1[i]["labels"][0] == V2[j]["labels"][0]:
                Gx.add_node((i,j))
    E1 = G1.edges(data='labels')
    E2 = G2.edges(data='labels')
    for e1 in E1:
        for e2 in E2:
            u1,v1,l1 = e1
            u2,v2,l2 = e2
            if l1 == l2 and Gx.has_node((u1,u2)) and Gx.has_node((v1,v2)):
                Gx.add_edge((u1,u2),(v1,v2))
    return Gx

class DirectProductGraphKernel:
    """
    Kernel counting random walks on the direct product graph
    """
    def __init__(self, lmbda: float):
        """
        Args:
            lmbda (float): coefficient such that lambda^k is the weight attributed to length k random walks
        """
        self.lmbda = lmbda

    def kernel(self,G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph) -> float:
        Gx = direct_product_graph(G1,G2)
        try:
            Ax = nx.adjacency_matrix(Gx).todense()
            n = len(Ax)
            return np.sum(np.linalg.inv(np.eye(n) - self.lmbda*Ax))
        except nx.NetworkXError:
            return 0


def WL_relabelling(G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph, h: int) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Performs the Weisfeiler-Lehman relabelling (cf "Weisfeiler-Lehman Graph Kernels", Nino Shervashidze, 2011 )
    Args:
        G1 (nx.classes.graph.Graph): Graph 1
        G2 (nx.classes.graph.Graph): Graph 2
        h (int): Depth of the relabelling method

    Returns:
        Tuple of the relabelled nodes at each depth between 0 and h.
    """

    V1 = np.array(list(nx.get_node_attributes(G1,"labels").values()))[:,0]
    V2 = np.array(list(nx.get_node_attributes(G2,"labels").values()))[:,0]
    multiset_labels1 = [V1]
    multiset_labels2 = [V2]
    n1 = len(V1)
    n2 = len(V2)
    for iter in range(1,h):
        if iter:
            m1, m2 = multiset_labels1[-1], multiset_labels2[-1]
            new_labels1 = []
            new_labels2 = []
            for node in range(n1):
                multi_set_label = []
                neighbors = G1.neighbors(node)
                for neigh in neighbors:
                    multi_set_label.append(m1[neigh])
                new_labels1.append(''.join(np.array([m1[node]]+sorted(multi_set_label)[::-1]).astype(str)))
            for node in range(n2):
                multi_set_label = []
                neighbors = G2.neighbors(node)
                for neigh in neighbors:
                    multi_set_label.append(m2[neigh])
                new_labels2.append(''.join(np.array([m2[node]]+sorted(multi_set_label)[::-1]).astype(str)))
        relabels1 = []
        relabels2 = []
        dico = {}
        c = 0
        for code in new_labels1:
            if not code in dico:
                dico[code] = c
                c += 1
        for code in new_labels2:
            if not code in dico:
                dico[code] = c
                c += 1
        for label in new_labels1:
            relabels1.append(dico[label])
        for label in new_labels2:
            relabels2.append(dico[label])
        multiset_labels1.append(relabels1)
        multiset_labels2.append(relabels2)
    return multiset_labels1, multiset_labels2

def WL_relabelling_with_edges(G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph, h: int) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Performs the Weisfeiler-Lehman relabelling adding edges labels in the relabelling procedure 
    (cf "Weisfeiler-Lehman Graph Kernels", Nino Shervashidze, 2011 ).
    Args:
        G1 (nx.classes.graph.Graph): Graph 1
        G2 (nx.classes.graph.Graph): Graph 2
        h (int): Depth of the relabelling method

    Returns:
        Tuple of the relabelled nodes at each depth between 0 and h.
    """
    V1 = np.array(list(nx.get_node_attributes(G1,"labels").values()))[:,0]
    V2 = np.array(list(nx.get_node_attributes(G2,"labels").values()))[:,0]
    multiset_labels1 = [V1]
    multiset_labels2 = [V2]
    n1 = len(V1)
    n2 = len(V2)
    for iter in range(1,h):
        if iter:
            m1, m2 = multiset_labels1[-1], multiset_labels2[-1]
            new_labels1 = []
            new_labels2 = []
            for node in range(n1):
                multi_set_label = []
                neighbors = dict(G1[node])
                for neigh, label_dico in neighbors.items():
                    multi_set_label.append(label_dico["labels"][0])
                    multi_set_label.append(m1[neigh])
                new_labels1.append(''.join(np.array([m1[node]]+sorted(multi_set_label)[::-1]).astype(str)))
            for node in range(n2):
                multi_set_label = []
                neighbors = dict(G2[node])
                for neigh, label_dico in neighbors.items():
                    multi_set_label.append(label_dico["labels"][0])
                    multi_set_label.append(m2[neigh])
                new_labels2.append(''.join(np.array([m2[node]]+sorted(multi_set_label)[::-1]).astype(str)))
        relabels1 = []
        relabels2 = []
        dico = {}
        c = 0
        for code in new_labels1:
            if not code in dico:
                dico[code] = c
                c += 1
        for code in new_labels2:
            if not code in dico:
                dico[code] = c
                c += 1
        for label in new_labels1:
            relabels1.append(dico[label])
        for label in new_labels2:
            relabels2.append(dico[label])
        multiset_labels1.append(relabels1)
        multiset_labels2.append(relabels2)
    return multiset_labels1, multiset_labels2

class WLKernel:
    """
    Implements kernel method using the WL relabelling method
    """
    def __init__(self, h: int, relabelling_method = WL_relabelling_with_edges, normalize: bool = True, lmbda: float = 1.2, edges_kernel= None):
        """
        Args:
            h (int): depth in the WL method
            relabelling_method (function, optional): Relabelling procedure
            normalize (bool, optional): if True divide the resulting kernel by the number of nodes. Defaults to True.
            lmbda (float, optional): weights to attribute to the successive depths. Defaults to 1.2.
            edges_kernel (_type_, optional): If not None add the all edges kernel to the result. Defaults to None.
        """
        self.h = h
        self.dirac = DiracKernel().kernel
        self.normalize = normalize
        self.lmbda = lmbda
        self.edges_kernel = edges_kernel
        self.relabelling_method = relabelling_method

    def kernel(self, G1: nx.classes.graph.Graph, G2: nx.classes.graph.Graph) -> float:
        m1, m2 = self.relabelling_method(G1, G2, self.h)
        s = 0
        for k, (L1, L2) in enumerate(zip(m1,m2)):
            s += self.lmbda**k * np.sum([np.sum([self.dirac(v1,v2) for v1 in L1]) for v2 in L2])
        if self.normalize:
            return s/(len(m1[0])+len(m2[0]))
        if self.edges_kernel is not None:
            s += self.edges_kernel(G1,G2)
        return s
    
