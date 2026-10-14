import xml.etree.ElementTree as ET
import pandas as pd
import random
from sklearn.model_selection import train_test_split

train_path = "data/NeuralTemporalRelation-EMNLP19/data/trainset-temprel.xml"
test_path = "data/NeuralTemporalRelation-EMNLP19/data/testset-temprel.xml"

label_mapping = {
    "BEFORE":0,
    "AFTER":1,
    "EQUAL":2,
    "VAGUE":3}

def data_processing_function(item):
    tokens = item.text.strip().split()
    words = []
    event1 = None
    event2 = None
    event1_index = None
    event2_index = None
    
    for i,token in enumerate(tokens):
        word,lemma,pos,marker = token.split("///")
        words.append(word)
    
        if marker == "E1":
            event1 = word
            event1_index = i

        if marker == "E2":
            event2 = word
            event2_index = i
    
    text = " ".join(words)

    return {
        "doc_id": item.attrib["DOCID"],
        "text": text,
        "event1": event1,
        "event2": event2,
        "event1_index": event1_index,
        "event2_index": event2_index,
        "relation": item.attrib["LABEL"],
        "label": label_mapping[item.attrib["LABEL"]],
        "sent_diff": int(item.attrib["SENTDIFF"])
    }

def load_matres_data(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    processed_data = []
    
    for item in root:
        processed_item = data_processing_function(item)
        processed_data.append(processed_item)

    dataframe = pd.DataFrame(processed_data)

    return dataframe

def split_train_val(dataframe):
    document_ids = dataframe["doc_id"].unique()
    train_ids,val_ids = train_test_split(document_ids,test_size = 20,random_state=123)

    train_data = dataframe[dataframe["doc_id"].isin(train_ids)].copy()
    val_data = dataframe[dataframe["doc_id"].isin(val_ids)].copy()

    return train_data, val_data
    

if __name__ == "__main__":

    train_val_data = load_matres_data(train_path)
    test_data = load_matres_data(test_path)
    #splitting training data to train and val data
    train_data,val_data = split_train_val(train_val_data)
    #count of event pairs 
    print("Train examples:",len(train_data))
    print("Val examples:",len(val_data))
    print("Test examples:",len(test_data))
    
    #print(train_data.iloc[0])
    train_data.to_csv("processed_data/train.csv",index=False)
    val_data.to_csv("processed_data/val.csv",index=False)
    test_data.to_csv("processed_data/test.csv",index=False)
    print("Dataset created")
    