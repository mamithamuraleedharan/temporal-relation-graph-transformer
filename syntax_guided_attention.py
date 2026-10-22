import torch
import torch.nn as nn


class SyntaxGuidedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(512,256)
        self.relation_embedding = nn.Embedding(num_embeddings = 46,embedding_dim = 32)
        self.relation_triple = nn.Linear(544,256)
        self.key = nn.Linear(256,256)
        self.value = nn.Linear(256,256)
        self.output = nn.Linear(256,256)

    def get_triple_representation(self,source,relation_id,target):
        relation_vector = self.relation_embedding(relation_id)
        combined_triple = torch.cat((source,relation_vector,target),dim=-1)
        triple_representation = self.relation_triple(combined_triple)
        return triple_representation

    def get_key_value(self,triple_representation):
        key = self.key(triple_representation)
        value = self.value(triple_representation)
        return key,value

        
    def forward(self,event1,event2,triples):
        combined_vector = torch.cat((event1,event2),dim = -1)
        query = self.query(combined_vector)
        query_heads = query.reshape(4, 64)
        keys = []
        values = []
        for source,relation_id,target in triples:
            triple_representation = self.get_triple_representation(
                source,relation_id,target)
            key,value = self.get_key_value(triple_representation)
            keys.append(key.reshape(4, 64))
            values.append(value.reshape(4, 64))
        key_heads = torch.stack(keys)
        value_heads = torch.stack(values)
        attention_scores = torch.sum(query_heads.unsqueeze(0) * key_heads,
        dim=-1) / 8.0
        attention_scores = attention_scores.transpose(0,1)
        attention_weights = torch.softmax(attention_scores,dim=-1)
        values_per_head = value_heads.transpose(0, 1)
        head_outputs = torch.sum(attention_weights.unsqueeze(-1) * values_per_head,dim=1)
        combined_heads = head_outputs.reshape(256)
        syntax_representation = self.output(combined_heads)
        return syntax_representation
        

if __name__ == "__main__":
    model = SyntaxGuidedAttention()
    event1 = torch.randn(256)
    event2 = torch.randn(256)

    triples = [
        (torch.randn(256), torch.tensor(4), torch.randn(256)),
        (torch.randn(256), torch.tensor(29), torch.randn(256)),
        (torch.randn(256), torch.tensor(4), torch.randn(256))]

    syntax_representation = model(event1, event2, triples)
    print("Syntax-guided representation:", syntax_representation.shape)