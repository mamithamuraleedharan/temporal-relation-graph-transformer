import spacy
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
 

if __name__ == "__main__":
    train_data = pd.read_csv("processed_data/train.csv")
    sample = train_data.iloc[0]
    text = sample["text"]
    edges,roots = build_dependency_graph(text)
    word_representations = get_combined_representations(text)
    initializer = NodeRepresentationInitializer()
    node_representations = initializer(word_representations)

    print("Number of graph nodes:", len(nlp(text)))
    print("Word representations:", word_representations.shape)
    print("Initial node representations:", node_representations.shape)