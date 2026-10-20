import spacy
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

        
    

if __name__ == "__main__":
    text = "John finished his work. He left the office."
    edges,roots = build_dependency_graph(text)
    print("Dependency edges:", edges)
    print("Root nodes:", roots)
    
    