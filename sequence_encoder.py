from bert import tokenizer,MODEL
from transformers import BertModel
import spacy
from spacy.tokens import Doc
from spacy.parts_of_speech import IDS
import torch
import torch.nn.functional as F

bert = BertModel.from_pretrained(MODEL)

nlp = spacy.load("en_core_web_sm")


def get_bert_embeddings(words):
    """
    Takes spaCy tokenized words and returns BERT contextual embeddings.
    """
    encoded = tokenizer(words,is_split_into_words= True,return_tensors="pt")
    
    device = next(bert.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device) 
    
    outputs = bert(
        input_ids=input_ids,
        attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state
    mapping = {}
    for i,word_id in enumerate(encoded.word_ids()):
        if word_id is None:
            continue
        if word_id not in mapping:
            mapping[word_id] = i


    word_embeddings = torch.stack([
    hidden_states[0, mapping[i], :]
    if i in mapping
    else torch.zeros(
        hidden_states.shape[-1],
        dtype=hidden_states.dtype,
        device=hidden_states.device
    )
    for i in range(len(words))
])
    
    return word_embeddings

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

def get_pos_one_hot(doc,pos_mapping):
    """
    Converts the POS tags of spaCy tokens into one-hot vectors.
    """
    pos_tags = get_pos_tags(doc)
    pos_ids = [pos_mapping[tag] for tag in pos_tags]
    pos_tensor = torch.tensor(pos_ids)
    pos_one_hot = F.one_hot(pos_tensor,num_classes=len(pos_mapping))
    return pos_one_hot

def get_combined_representations(text):
    doc = nlp(text)
    words = [token.text for token in doc]
    contextual_embeddings = get_bert_embeddings(words)
    pos_mapping = create_pos_mapping()
    pos_one_hot = get_pos_one_hot(doc,pos_mapping)
    pos_one_hot = pos_one_hot.to(contextual_embeddings.device)
    combined_embeddings = torch.cat((contextual_embeddings, pos_one_hot.float()),dim=1)
    return combined_embeddings

def align_event_to_spacy(text,event_index,doc):
    """
    Converts a MATRES event index into its corresponding spaCy token index.
    """
    words = text.split(" ")
    char_position = 0
    for i in range(event_index):
        char_position += len(words[i]) + 1
    for token in doc:
        if token.idx == char_position:
            return token.i
    raise ValueError("Event could not be aligned to spaCy tokens")
    
