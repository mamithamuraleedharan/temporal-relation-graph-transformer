# Temporal Relation Extraction
Temporal Relation Extraction is the task of identifying temporal relationships between events in text. One major challenge is determining temporal relationships between events that are far apart or embedded in complex grammatical structures. This study investigates whether using syntactic dependency information can improve temporal relation classification by comparing a BERT baseline model with a Syntax-Guided Graph Transformer (SGT) on the MATRES dataset. The study also examines linear event distance and dependency path length.

This project implements a simplified version of the paper Extracting Temporal Event Relation with Syntax-Guided Graph Transformer

Link : https://aclanthology.org/2022.findings-naacl.29/

# Dataset 
The study uses a processed version of the original MATRES dataset for temporal relation classification. The MATRES dataset contains training and test sets in XML format. The XML files were obtained from the NeuralTemporalRelation-EMNLP19 repository

Link : https://github.com/qiangning/NeuralTemporalRelation-EMNLP19

# Requirements
The project is implemented in Python and uses the following libraries:
- PyTorch
- Transformers(Hugging Face)
- spaCy(en_core_web_sm)
- NetworkX
- pandas
- NumPy
- scikit-learn
- Pretrained language model: google-bert/bert-base-uncased

# Project Structure

The repository is organized as follows:

| File / Folder | Description |
|---|---|
| `processed_data/` | Contains the preprocessed MATRES dataset in CSV format, including training, validation, and test sets. |
| `data_preprocessing.py` | Extracts event pairs and labels from the XML data and creates the CSV files. |
| `bert.py` | Defines the BERT baseline model. |
| `sgt_model.py` | Main SGT model that connects the sequence encoder, dependency graph, graph attention, syntax-guided attention, node fusion, and temporal relation classifier. |
| `train_bert.py` / `evaluate_bert.py` | Scripts for training and evaluating the BERT baseline. |
| `train_sgt.py` / `evaluate_sgt.py` | Scripts for training and evaluating the SGT model. |
| `SGT_Presentation_Demo.ipynb` | Demonstrates the model architecture, predictions, and experimental results. |
| `logs/` | Contains training and evaluation logs. |

# Findings
- BERT achieved higher accuracy, while SGT achieved better macro F1.
- SGT showed better performance on minority classes, especially VAGUE.
- Both models struggled as event distance and dependency-path length increased.

# Limitations
- Class imbalance: Both models struggled to predict EQUAL.
- Overfitting: Training loss decreased while validation loss increased.
- Simplified SGT implementation: Some components differ from the original paper.

# Reference 
Shuaicheng Zhang, Qiang Ning, and Lifu Huang. 2022. Extracting Temporal Event Relation with Syntax-guided Graph Transformer. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 379–390, Seattle, United States. Association for Computational Linguistics.