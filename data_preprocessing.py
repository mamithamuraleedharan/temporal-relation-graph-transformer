import xml.etree.ElementTree as ET
import pandas as pd

train_path = "data/NeuralTemporalRelation-EMNLP19/data/trainset-temprel.xml"
test_path = "data/NeuralTemporalRelation-EMNLP19/data/testset-temprel.xml"

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
    

if __name__ == "__main__":
    train_data = load_matres_data(train_path)
    test_data = load_matres_data(test_path)
    print("Train shape:",train_data.shape)
    print("Test shape:",test_data.shape)
    