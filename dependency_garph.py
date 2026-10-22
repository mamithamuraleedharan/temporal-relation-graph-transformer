import spacy
import torch
import pandas as pd
import torch.nn as nn
import networkx as nx
from sequence_encoder import get_combined_representations, align_event_to_spacy
from graph_attention import GraphMultiheadAttention
from syntax_guided_attention import SyntaxGuidedAttention

nlp = spacy.load("en_core_web_sm")

def build_dependency_graph(text):
    edges = []
    roots = []
    doc = nlp(text)
    for token in doc:
        if token.dep_ == "ROOT":
            roots.append(token.i)
            continue
        edges.append((token.i,token.dep_,token.head.i)) # (vi,rij,vj) where v1 = source node, rij = dependency relation label, vj = destination node
    for i in range(len(roots) - 1):
        edges.append((roots[i], "cross_sentence", roots[i + 1]))
    return edges,roots

class NodeRepresentationInitializer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(788,256)
        
    def forward(self,x):
        x = self.hidden(x)
        return x
        
def create_adjacency_matrix(num_nodes,edges):
    """
    Converts dependency edges into adjacency matrix.
    """
    adj_matrix = torch.zeros((num_nodes,num_nodes),dtype=torch.bool)
    adj_matrix.fill_diagonal_(True)
    for source,relation,target in edges:
        adj_matrix[source,target] = True
        adj_matrix[target,source] = True
    return adj_matrix

def get_dependency_path(edges,event1_index,event2_index):
    G = nx.Graph()
    for source,relation,target in edges:
        G.add_edge(source,target)
    path = nx.shortest_path(G,source=event1_index,target=event2_index)
    return path

def get_path_triples(edges,path):
    dep_path_nodes = set(path)
    dep_triples = []
    for source,relation,target in edges:
        if source in dep_path_nodes or target in dep_path_nodes:
            dep_triples.append((source,relation,target))
    return dep_triples
    
def create_dependency_mapping():
    """
    Creates a fixed mapping of dependency relation labels to integer IDs
    """
    dependency_mapping = {}
    dependency_labels = sorted(nlp.get_pipe("parser").labels)
    if "cross_sentence" not in dependency_labels:
        dependency_labels.append("cross_sentence")
    for i,label in enumerate(dependency_labels):
        dependency_mapping[label] = i
    return dependency_mapping

def prepare_dependency_triples(triples,node_representation,dependency_mapping):
    prepared_triples = []
    for source,relation,target in triples:
        source_vector = node_representation[source]
        target_vector = node_representation[target]
        relation_id = dependency_mapping[relation]
        relation_tensor = torch.tensor(relation_id,dtype=torch.long,device=node_representation.device)
        prepared_triples.append((source_vector,relation_tensor, target_vector))
    return prepared_triples
        
