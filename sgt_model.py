import torch
import torch.nn as nn
import pandas as pd
from sequence_encoder import get_combined_representations,align_event_to_spacy,bert
from graph_attention import GraphMultiheadAttention
from dependency_garph import nlp,NodeRepresentationInitializer,create_adjacency_matrix,build_dependency_graph, get_dependency_path,get_path_triples,create_dependency_mapping,prepare_dependency_triples
from syntax_guided_attention import SyntaxGuidedAttention
from node_fusion import NodeRepresentationFusion
from temporal_classifier import TemporalRelationClassifier

class SGTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert
        self.node_initializer = NodeRepresentationInitializer()
        self.graph_attention = GraphMultiheadAttention()
        self.syntax_attention = SyntaxGuidedAttention()
        self.node_fusion = NodeRepresentationFusion()
        self.temporal_classifier = TemporalRelationClassifier()

    def forward(self,word_representations,edges,event1_index,event2_index):
        node_representations = self.node_initializer(word_representations)
        num_nodes = node_representations.shape[0]
        adj_matrix = create_adjacency_matrix(num_nodes,edges)
        attention_mask = torch.logical_not(adj_matrix)
        attention_mask = attention_mask.to(node_representations.device)
        batched_nodes = node_representations.unsqueeze(0)
        graph_output,attention_weights = self.graph_attention(batched_nodes,attention_mask)
        graph_output = graph_output.squeeze(0)
        path = get_dependency_path(edges,event1_index,event2_index)
        triples = get_path_triples(edges,path)
        dependency_mapping = create_dependency_mapping()
        prepared_triples = prepare_dependency_triples(triples,node_representations, dependency_mapping)
        event1 = node_representations[event1_index]
        event2 = node_representations[event2_index]
        syntax_representation = self.syntax_attention(event1,event2,prepared_triples)
        fused_representations = self.node_fusion(graph_output,node_representations, event1_index,event2_index,syntax_representation)
        event1_final = fused_representations[event1_index]
        event2_final = fused_representations[event2_index]
        logits = self.temporal_classifier(event1_final,event2_final)
        return logits

