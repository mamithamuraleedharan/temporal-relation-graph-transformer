import torch
import torch.nn as nn


class TemporalRelationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(512,4)

    def forward(self,event1,event2):
        combined_vector = torch.cat((event1,event2),dim = -1)
        logits = self.classifier(combined_vector)
        return logits

