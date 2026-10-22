import pandas as pd
import torch
import torch.nn as nn
from bert import BertBaseline, find_bert_index
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import time
from tqdm.auto import tqdm

TRAIN_PATH = "processed_data/train.csv"
VAL_PATH = "processed_data/val.csv"

# creating custom dataset. 3 functions :  __init__, __len__, and __getitem__
class MatresDataset(Dataset):
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
        encoded,bert_event1_index,bert_event2_index = find_bert_index(text,event1_index,event2_index)
        return {
            "input_ids":encoded["input_ids"].squeeze(0),
            "attention_mask":encoded["attention_mask"].squeeze(0),
            "event1_index":torch.tensor(bert_event1_index),
            "event2_index":torch.tensor(bert_event2_index),
            "label":torch.tensor(label)
                }

def train_one_epoch(model,train_loader,loss_function,optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        event1_index = batch["event1_index"].to(device)
        event2_index = batch["event2_index"].to(device)
        labels = batch["label"].to(device)
        
        pred = model(input_ids,attention_mask,event1_index,event2_index)
        loss = loss_function(pred,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss = total_loss+loss.item()
        progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

    average_loss = total_loss/len(train_loader)
    return average_loss
            
def validate(model,val_loader,loss_function,device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            event1_index = batch["event1_index"].to(device)
            event2_index = batch["event2_index"].to(device)
            labels = batch["label"].to(device)
            pred = model(input_ids,attention_mask,event1_index,event2_index)
            loss = loss_function(pred,labels)
            total_loss = total_loss+loss.item()
            predicted_labels = torch.argmax(pred,dim=1)
            for i in range(len(labels)):
                if predicted_labels[i] == labels[i]:
                    correct += 1

                total += 1

    average_loss = total_loss/len(val_loader)
    accuracy = correct / total
    return average_loss,accuracy    
    


if __name__ == "__main__":
    train_dataset = MatresDataset(TRAIN_PATH)
    val_dataset = MatresDataset(VAL_PATH)
    
    # example = train_dataset[0]
    #creating data loaders
    train_loader = DataLoader(train_dataset,batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=8,shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model = BertBaseline().to(device)
    #loss function
    loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),lr=0.00002)
    print("Training examples:",len(train_dataset))
    print("Validation examples:",len(val_dataset))

    epochs = 10
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}", flush=True)
        start_time = time.time()
        train_loss = train_one_epoch(model,train_loader,loss,optimizer,device)
        val_loss,val_accuracy = validate(model,val_loader,loss,device)
        stop_time = time.time()-start_time
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Time: {stop_time / 60:.2f} minutes")

        if val_accuracy>best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(),"models/bert_best_256.pt")
            print("New best model saved!")
        print("----------------------")

    