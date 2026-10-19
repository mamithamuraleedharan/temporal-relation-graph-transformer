# Temporal Relation Extraction
Temporal Relation Extraction is the task of understanding the timeline
of events in text. Here we are reproducing and extending the paper Extracting Temporal Event Relation with Syntax-guided Graph Transformer. It becomes difficult for the model to determine the temporal relationship between events when they are far apart in a text or when they have complex grammatical structure. To address this issue, the authors introduce a syntax guided graph transformer (SGT) and see if we can really improve temporal extraction by using syntactic connection between
events.
# Dataset 
We use the MATRES dataset, which contains annotated event pairs and their temporal relationships.The dataset contains four relation labels: BEFORE, AFTER, EQUAL, and VAGUE.
# Files
1. data_preprocessing.py - Loads and preprocesses the MATRES dataset, extracts event pairs and creates training, validation and test CSV files.
2. bert.py - Defines the BERT baseline model for temporal relation classification.
3. train_bert.py - Trains the BERT baseline, validates its performance and saves the best model checkpoint.
4. evaluate_bert.py- Evaluates the trained BERT model using accuracy, precision,recall,F1-score and a confusion matrix.
5. dependency_parser.py - Constructs dependency graphs using spaCy, aligns graph nodes with BERT tokens