import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from bert import BertBaseline
from train_bert import MatresDataset

VAL_PATH = "processed_data/val.csv"
TEST_PATH = "processed_data/test.csv"

def load_model(device):
    model = BertBaseline()
    model.load_state_dict(torch.load("models/bert_best.pt", map_location=device,weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def evaluate(model,data_loader,device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            event1_index = batch["event1_index"].to(device)
            event2_index = batch["event2_index"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids,attention_mask,event1_index, event2_index)
            predicted_labels = torch.argmax(outputs,dim=1)
            for i in range(len(labels)):
                predictions.append(predicted_labels[i].item())
                actual_labels.append(labels[i].item())
    return actual_labels,predictions

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = MatresDataset(TEST_PATH)
    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=False)
    model = load_model(device)
    actual_labels,predictions = evaluate(model,test_loader,device)
    print("Test Accuracy:",accuracy_score(actual_labels, predictions))
    print("\nClassification Report:")
    print(classification_report(actual_labels,predictions,
        labels=[0, 1, 2, 3],
        target_names=["BEFORE","AFTER","EQUAL","VAGUE"],
        zero_division=0
    ))
    print("\nConfusion Matrix:")
    print(confusion_matrix(actual_labels,predictions,labels=[0,1,2,3]))
    