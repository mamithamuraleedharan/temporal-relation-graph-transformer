from transformers import AutoTokenizer,BertModel
import torch
import torch.nn as nn

MODEL = "google-bert/bert-base-uncased"

tokenizer =  AutoTokenizer.from_pretrained(MODEL,use_fast=True)

def find_bert_index(text,event1_index,event2_index):
    words = text.split()
    encoded = tokenizer(words,is_split_into_words=True, truncation=True,max_length=256,padding="max_length",return_tensors="pt")
    word_ids = encoded.word_ids()
    bert_event1_index = None
    bert_event2_index = None  
    for i,word_id in enumerate(word_ids):
        if word_id == event1_index and bert_event1_index is None:
            bert_event1_index = i
        if word_id == event2_index and bert_event2_index is None:
            bert_event2_index = i
    if bert_event1_index is None or bert_event2_index is None:
        raise ValueError("Error")
    return encoded, bert_event1_index, bert_event2_index
    


class BertBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL)
        hidden_dim_size = self.bert.config.hidden_size # gets dimension size automatically
        self.classifier = nn.Linear(hidden_dim_size*2,4)

    def forward(self,input_ids,attention_mask,event1_index,event2_index):
        outputs = self.bert(input_ids = input_ids,attention_mask = attention_mask)
        hidden_states = outputs.last_hidden_state
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)

        event1_vector = hidden_states[batch_indices,event1_index,:]
        event2_vector = hidden_states[batch_indices,event2_index,:]
        combined = torch.cat((event1_vector, event2_vector), dim=-1)
        logits = self.classifier(combined)
        return logits

