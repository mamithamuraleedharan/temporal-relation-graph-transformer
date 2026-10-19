import spacy
from spacy.tokens import Doc
from bert import tokenizer
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
    for i in range(len(roots) - 1):
        edges.append((roots[i], "cross_sentence", roots[i + 1]))
    return doc,edges
    
        
def align_graph_to_bert(text):
    mapping = {}
    words = text.split()
    encoded = tokenizer(words,is_split_into_words=True, truncation=True,max_length=256,padding="max_length",return_tensors="pt")
    word_ids = encoded.word_ids()
    for i,word_id in enumerate(word_ids):
        if word_id is None:
            continue
        elif word_id not in mapping:
            mapping[word_id] = i
    return mapping

        
