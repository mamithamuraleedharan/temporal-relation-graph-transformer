import spacy
import pandas as pd
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")


def build_dependency_graph(text):
    #sentence = nlp("John finished his work before he left")
    words = text.split()
    doc = Doc(nlp.vocab, words=words)
    doc = nlp(doc)
    edges = []
    roots = []
    for token in doc:
        if token.dep_ == "ROOT":
            roots.append(token.i)
            continue
       
        edges.append((token.i,token.dep_,token.head.i)) # (vi,rij,vj) where v1 = source node, rij = dependency relation label, vj = destination node
    if len(roots) == 2:
        edges.append((roots[0], "cross_sentence", roots[1]))
    return doc,edges
    
def check_tokenization_mismatch(dataframe):
    mismatch = 0
    for index,row in dataframe.iterrows():
        text = row["text"]
        doc,edges = build_dependency_graph(text)
        spacy_tokens = [token.text for token in doc]
        matres_tokens = text.split()
        if matres_tokens != spacy_tokens:
            mismatch +=1
    return mismatch
        
        

if __name__ == "__main__":
    data = pd.read_csv("processed_data/train.csv")
    doc,edges =build_dependency_graph(data)
   