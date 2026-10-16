# Temporal Relation Extraction
Temporal Relation Extraction is the task of understanding the timeline
of events in text. Here we are reproducing and extending the paper Extracting Temporal Event Relation with Syntax-guided Graph Transformer. It becomes difficult for the model to determine the temporal relationship between events when they are far apart in a text or when they have complex grammatical structure. To address this issue, the authors introduce a syntax guided graph transformer (SGT) and see if we can really improve temporal extraction by using syntactic connection between
events.
# Files
1. data_preprocessing.py - Prepares MATRES data
2. bert.py - Defining BERT model
3. train_bert.py - trains and validates the model.