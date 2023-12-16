import os
import json
import math
from pathlib import Path
from datetime import datetime

import torch
import evaluate
import numpy as np
import pandas as pd

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, get_scheduler

# assert torch.cuda.is_available()

class GeneDataset(Dataset):
    """
    Constructs Gene Dataset with the given data
    Returns unmasked and masked data
    Random mask comes with bias towards nonzero expression values
    """
    
    def __init__(self, source, mask_id, num_bins, mask_ratio, nonzero_ratio):
        self.data = source
        self.mask_id = mask_id
        self.num_bins = num_bins
        self.mask_ratio = mask_ratio
        self.nonzero_ratio = nonzero_ratio
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        
        truth = torch.tensor(self.data[index], dtype=torch.long)
        mask_num = (int)(truth.shape[0] * self.mask_ratio)
        
        nonzero_indices = (truth % self.num_bins).nonzero().view(-1)
        zero_indices = (truth % self.num_bins == 0).nonzero().view(-1)
        nonzero_indices = np.random.choice(
            nonzero_indices, 
            int(min(mask_num * self.nonzero_ratio, nonzero_indices.shape[0])), 
            replace=False)
        zero_indices = np.random.choice(
            zero_indices, 
            int(min(mask_num * (1.0 - self.nonzero_ratio), zero_indices.shape[0])),
            replace=False)
        
        masked_truth = torch.tensor(self.data[index], dtype=torch.long)
        masked_truth[zero_indices] = self.mask_id
        masked_truth[nonzero_indices] = self.mask_id
        
        return masked_truth, truth
    
class GeneClassificationModel(torch.nn.Module):
    
    def __init__(self, mask_id, model_hidden_size, model_hidden_layers, model_attention_heads, num_genes):
        super().__init__()
        self.vocab_size = num_genes
        self.hidden_size = model_hidden_size
        self.num_hidden_layers = model_hidden_layers
        self.num_attention_heads = model_attention_heads
        self.Bert = BertModel(BertConfig(
            vocab_size = mask_id + 1,
            hidden_size=model_hidden_size,
            num_hidden_layers=model_hidden_layers,
            num_attention_heads=model_attention_heads,
            intermediate_size=model_hidden_size * 4,
            max_position_embeddings = num_genes
        ))
        self.ClassificationHead = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.LayerNorm(model_hidden_size, eps=1e-12),
            torch.nn.Dropout(0.1, inplace=False),
            torch.nn.Linear(model_hidden_size, mask_id + 1)
        )
        
    def forward(self, x):
        hidden = self.Bert(x)
        return self.ClassificationHead(hidden.last_hidden_state).squeeze()

def train(config, train_dataloader, test_dataloader, model, optimizer, criterion, mask_id):
    
    exp_id = np.random.randint(0, 10000)
    
    Path(f"./saved_models/random_mask_bias/exp_{exp_id}/").mkdir(parents=True, exist_ok=True)
    Path(f"./training_logs/random_mask_bias/").mkdir(parents=True, exist_ok=True)
    log = open(f"./training_logs/random_mask_bias/exp_{exp_id}.log", "x")
    
    log.write(str(exp_id))
    log.flush()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    num_training_steps = config['num_epochs'] * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=config['scheduler'],
        optimizer=optimizer,
        num_warmup_steps=config['num_warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    for epoch in range(config['num_epochs']):
        total_train_loss = 0  
        model.train()
        log.write(f"Epoch {epoch}, total step {len(train_dataloader)}\n")
        log.write(f"Begin training, {datetime.now()}\n")
        log.flush()
        
        for batch, (x, y) in enumerate(train_dataloader):
            
            x = x.to(device)
            y = y.to(device).view(-1)
            
            outputs = model(x)
            outputs = outputs.view(-1, outputs.shape[-1])
            loss = criterion(outputs, y)
            loss.backward()
            total_train_loss += loss
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if batch % 200 == 0:
                predictions = torch.argmax(torch.nn.functional.log_softmax(outputs, dim=-1), dim=-1)
                
                x_flat = x.view(-1)
                masked = (x_flat == mask_id)
                indices = torch.nonzero(masked)
                predictions = predictions.view(-1)[indices]
                references = y.view(-1)[indices]
        
                metric = evaluate.load("accuracy")
                metric.add_batch(predictions=predictions.view(-1), references=references.view(-1))
                log.write(f"Batch {batch}, loss = {loss}, accuracy = {metric.compute()['accuracy']}\n")
                log.flush()
            
        model.eval()
        metric = evaluate.load("accuracy")
        total_eval_loss = 0
        log.write(f"Begin evaluating, {datetime.now()}\n")
        log.flush()
    
        for batch, (x, y) in enumerate(test_dataloader):
            
            x = x.to(device)
            y = y.to(device).view(-1)
            
            with torch.no_grad():
                outputs = model(x)
                outputs = outputs.view(-1, outputs.shape[-1])
            
            loss = criterion(outputs, y)
            predictions = torch.argmax(torch.nn.functional.log_softmax(outputs, dim=-1), dim=-1)
            
            x_flat = x.view(-1)
            masked = (x_flat == mask_id)
            indices = torch.nonzero(masked)
            predictions = predictions.view(-1)[indices]
            references = y.view(-1)[indices]
            
            metric.add_batch(predictions=predictions.view(-1), references=references.view(-1))
            total_eval_loss += loss
            
            if batch % 200 == 0:
                log.write(f"Batch {batch}, loss = {loss}, accuracy = {metric.compute()['accuracy']}\n")
                log.flush()

        results = metric.compute()
        log.write("Evaluation complete\n")
        log.write(str(results))
        log.write("\n")
        log.write(str(total_eval_loss / len(test_dataloader)))
        log.write("\n")
        log.flush()
    
        torch.save(model.state_dict(), f"./saved_models/random_mask_bias/exp_{exp_id}/{model.hidden_size}_{model.num_hidden_layers}_{model.num_attention_heads}_epoch_{epoch}_loss_{(float)(total_eval_loss / len(train_dataloader))}")
        
def main():
    
    with open("./settings.json") as f:
        config = json.loads(f.read())
    
    train_data = np.load("./processed_data/train_data.npy")
    test_data = np.load("./processed_data/valid_data.npy")
    
    num_bins = np.max([np.max(train_data), np.max(test_data)]) + 1
    num_genes = train_data.shape[1]
    mask_id = num_bins * num_genes
    
    # print(num_bins, num_genes, mask_id)
    
    for i in range(num_genes):
        train_data[:, i] += num_bins * i
        test_data[:, i] += num_bins * i
        
    train_dataset = GeneDataset(train_data, mask_id, num_bins, config["mask_ratio"], config["nonzero_ratio"])
    test_dataset = GeneDataset(test_data, mask_id, num_bins, config["mask_ratio"], config["nonzero_ratio"])

    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["train_batch_size"], shuffle=True)

    model = GeneClassificationModel(
        mask_id,
        config["model_hidden_size"],
        config["model_hidden_layers"],
        config["model_attention_heads"],
        num_genes
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    
    train(config, train_dataloader, test_dataloader, model, optimizer, criterion, mask_id)
       
if __name__ == "__main__":
    main()