import os
import time
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset,DataLoader,Subset
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sequence_encoder import get_combined_representations,align_event_to_spacy
from dependency_garph import nlp,build_dependency_graph
from sgt_model import SGTModel

embedding_cache = {}

class SGTDataset(Dataset):
    def __init__(self,path):
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        row = self.data.iloc[index]
        text = row["text"]
        event1_index = int(row["event1_index"])
        event2_index = int(row["event2_index"])
        label = int(row["label"])
        return {
            "text": text,
            "event1_index": event1_index,
            "event2_index": event2_index,
            "label": label}

def prepare_sgt_sample(sample):
    text = sample["text"]
    event1_index = sample["event1_index"]
    event2_index = sample["event2_index"]
    label = sample["label"]
    doc = nlp(text)

    event1_spacy_index = align_event_to_spacy(text,event1_index,doc)  
    event2_spacy_index = align_event_to_spacy(text,event2_index,doc)
    
    word_representations = get_combined_representations(text)
    edges,roots = build_dependency_graph(text)

    return {
    "word_representations": word_representations,
    "edges": edges,
    "event1_index": event1_spacy_index,
    "event2_index": event2_spacy_index,
    "label": label}

def train_one_epoch(model,train_loader,loss_function,optimizer,device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        sample = {
            "text": batch["text"][0],
            "event1_index": batch["event1_index"][0].item(),
            "event2_index": batch["event2_index"][0].item(),
            "label": batch["label"][0].item()}
        prepared = prepare_sgt_sample(sample)
        word_representations = prepared["word_representations"].to(device)
        edges = prepared["edges"]
        event1_index = prepared["event1_index"]
        event2_index = prepared["event2_index"]
        label = torch.tensor(prepared["label"],dtype=torch.long,device=device)
        logits = model(word_representations,edges,event1_index,event2_index)
        loss = loss_function(logits.unsqueeze(0),label.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

    average_loss = total_loss / len(train_loader)
    return average_loss

def validate(model,val_loader,loss_function,device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in val_loader:
            sample = {
                "text": batch["text"][0],
                "event1_index": batch["event1_index"][0].item(),
                "event2_index": batch["event2_index"][0].item(),
                "label": batch["label"][0].item()}
            prepared = prepare_sgt_sample(sample)
            word_representations = prepared["word_representations"].to(device)
            edges = prepared["edges"]
            event1_index = prepared["event1_index"]
            event2_index = prepared["event2_index"]           
            label = torch.tensor(prepared["label"],dtype=torch.long,device=device)
            logits = model(word_representations,edges,event1_index,event2_index)
            loss = loss_function(logits.unsqueeze(0),label.unsqueeze(0))
            total_loss += loss.item()
            predicted_label = torch.argmax(logits).item()
            if predicted_label == label.item():
                correct += 1           
            total += 1
            predictions.append(predicted_label)
            actual_labels.append(label.item())
    average_loss = total_loss / len(val_loader)
    accuracy = correct / total
    macro_f1 = f1_score(actual_labels,predictions,labels=[0, 1, 2, 3],
    average="macro",zero_division=0)
    return average_loss,accuracy,macro_f1
            

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_dataset = SGTDataset("processed_data/train.csv")
    val_dataset = SGTDataset("processed_data/val.csv")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SGTModel().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW([
        {
            "params": model.bert.parameters(),
            "lr": 1e-5
        },
        {
            "params": [
                param for name, param in model.named_parameters()
                if not name.startswith("bert.")
            ],
            "lr": 1e-4
        }
    ])

    epochs = 5
    best_val_macro_f1 = -1.0
    print("Device:", device)
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Epochs:", epochs)
    print("BERT learning rate:", optimizer.param_groups[0]["lr"])
    print("SGT learning rate:", optimizer.param_groups[1]["lr"])
    for epoch in range(epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}", flush=True)

        train_loss = train_one_epoch(model,train_loader,loss_function, optimizer,device)
        val_loss,val_accuracy,val_macro_f1 = validate(model,val_loader, loss_function,device)
        elapsed = time.time() - start_time

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro F1: {val_macro_f1:.4f}")
        print(f"Epoch time: {elapsed / 60:.2f} minutes")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

            torch.save(model.state_dict(), "models/sgt_bert_finetuned_best.pt")
            print("New best fine-tuned SGT model saved!")
        print("----------------------", flush=True)
