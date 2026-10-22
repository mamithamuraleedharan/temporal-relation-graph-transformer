import torch
import torch.nn as nn

class NodeRepresentationFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_layer = nn.Linear(512,256)
        self.non_event_projection = nn.Linear(256,256)
        self.layer_norm = nn.LayerNorm(256)


    def forward(self, graph_representations, previous_representations,
            event1_index, event2_index, syntax_representation):
        graph_event1 = graph_representations[event1_index]
        graph_event2 = graph_representations[event2_index]
        combined_event1= torch.cat((graph_event1,syntax_representation),dim = -1)
        fused_event1 = self.fusion_layer(combined_event1)
        combined_event2= torch.cat((syntax_representation,graph_event2),dim = -1)
        fused_event2 = self.fusion_layer(combined_event2)
        updated_representations = self.non_event_projection(graph_representations)
        updated_representations = updated_representations.clone()
        updated_representations[event1_index] = fused_event1
        updated_representations[event2_index] = fused_event2
        residual_output = updated_representations+previous_representations
        normalized_output = self.layer_norm(residual_output)
        return normalized_output


