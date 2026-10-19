import spacy
import pandas as pd
from spacy.tokens import Doc
from bert import tokenizer
from spacy.parts_of_speech import IDS
nlp = spacy.load("en_core_web_sm")

def find_boundary_candidates(text,event1_index,event2_index):
    words = text.split()
    candidates = []
    for i in range(event1_index + 1,event2_index):
        if words[i] in [".","?","!"]:
            next_index = i + 1
            while next_index < len(words) and words[next_index] in ['"',"``","''"]:
                next_index += 1
            if next_index < len(words):
                candidates.append(i + 1)
    return candidates
    

def build_dependency_graph(text,event1_index,event2_index,sent_diff):
    #sentence = nlp("John finished his work before he left")
    words = text.split()
    doc = Doc(nlp.vocab,words=words)
    if sent_diff == 0:
        for token in doc:
            token.is_sent_start = False
        doc[0].is_sent_start = True
    elif sent_diff == 1:
        candidates = find_boundary_candidates(text,event1_index, event2_index)
        if len(candidates) == 1:
            for token in doc:
                token.is_sent_start = False
            doc[0].is_sent_start = True
            doc[candidates[0]].is_sent_start = True
            
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

def get_pos_tags(doc):
    pos_tags = []
    for token in doc:
        pos_tags.append(token.pos_)
    return pos_tags

def create_pos_mapping(): 
    pos_mapping = {"PAD": 0,"UNK": 1}   
    pos_tags = sorted(tag for tag in IDS if tag != "")
    for i,tag in enumerate(pos_tags, start=2):
        pos_mapping[tag] = i    
    return pos_mapping         

if __name__ == "__main__":
    train_data = pd.read_csv("processed_data/train.csv")
    val_data = pd.read_csv("processed_data/val.csv")
    test_data = pd.read_csv("processed_data/test.csv")
    print("Train mismatches:", check_sentence_boundaries(train_data))
    print("Validation mismatches:", check_sentence_boundaries(val_data))
    print("Test mismatches:", check_sentence_boundaries(test_data))