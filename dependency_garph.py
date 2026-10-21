import spacy
import torch
import pandas as pd
import torch.nn as nn
from sequence_encoder import get_combined_representations

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
        
if __name__ == "__main__":
    text = "John finished his work before he left."
    edges,roots = build_dependency_graph(text)
    num_nodes = len(nlp(text))
    adj_matrix = create_adjacency_matrix(num_nodes, edges)
    attention_mask = torch.logical_not(adj_matrix)
    print("Adjacency matrix:")
    print(adj_matrix.int())
    print("\nAttention mask:")
    print(attention_mask.int())