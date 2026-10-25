import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

from sgt_model import SGTModel
from train_sgt import SGTDataset,prepare_sgt_sample

TEST_PATH = "processed_data/test.csv"

def load_model(device):
    model = SGTModel()
    model.load_state_dict(torch.load("models/sgt_bert_finetuned_best.pt", map_location=device,weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def evaluate(model,data_loader,device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            sample = {
                "text": batch["text"][0],
                "event1_index": batch["event1_index"][0].item(),
                "event2_index": batch["event2_index"][0].item(),
                "label": batch["label"][0].item()
            }
            prepared = prepare_sgt_sample(sample)
            word_representations = prepared["word_representations"].to(device)
            edges = prepared["edges"]
            event1_index = prepared["event1_index"]
            event2_index = prepared["event2_index"]           
            logits = model(word_representations,edges,event1_index,event2_index)            
            predicted_label = torch.argmax(logits).item()
            predictions.append(predicted_label)
            actual_labels.append(prepared["label"])
    return actual_labels,predictions



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = SGTDataset(TEST_PATH)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)
    model = load_model(device)
    actual_labels,predictions = evaluate(model,test_loader,device)

    print("Test examples:", len(actual_labels))
    print("Test Accuracy:", accuracy_score(actual_labels, predictions))

    print("\nClassification Report:")
    print(classification_report(actual_labels,predictions,
        labels=[0, 1, 2, 3],
        target_names=["BEFORE", "AFTER", "EQUAL", "VAGUE"],
        zero_division=0))

    print("\nConfusion Matrix:")
    
    print(confusion_matrix(
        actual_labels,
        predictions,
        labels=[0, 1, 2, 3]
    ))
