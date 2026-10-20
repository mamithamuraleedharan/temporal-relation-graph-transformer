from bert import tokenizer,MODEL
from transformers import BertModel
import spacy
from spacy.tokens import Doc
from spacy.parts_of_speech import IDS
import torch
import torch.nn.functional as F

bert = BertModel.from_pretrained(MODEL)
bert.eval()
nlp = spacy.load("en_core_web_sm")


def get_bert_embeddings(text):
    """
    Get a sentence. tokenize them and pass the tokens through bert. Return contexual embeddings
    """
    words = text.split()
    encoded = tokenizer(words,is_split_into_words= True,return_tensors="pt")
    outputs = bert(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"])
    return outputs.last_hidden_state

def get_pos_tags(doc):
    """
    Extracts the pos tags of words in sentence
    """
    pos_tags = []
    for token in doc:
        pos_tags.append(token.pos_)
    return pos_tags

def create_pos_mapping():
    """
    Creates a fixed mapping of POS tags to consecutive integer IDs.
    """
    pos_mapping = {}
    pos_tags = []
    for tag in IDS:
        if tag != "":
            pos_tags.append(tag)
    pos_tags = sorted(pos_tags)
    for i,tag in enumerate(pos_tags):
        pos_mapping[tag] = i
    return pos_mapping


if __name__ == "__main__":
    text = "John finished his work before he left."
    #contextual_embeddings = get_bert_embeddings(text)
    words = text.split()
    doc = nlp(text)
    pos_tags = get_pos_tags(doc)
    pos_mapping = create_pos_mapping()
    pos_ids = [pos_mapping[tag] for tag in pos_tags]
    pos_tensor = torch.tensor(pos_ids)
    pos_one_hot = F.one_hot(pos_tensor,num_classes=len(pos_mapping))
    print(pos_one_hot.shape)
    
    