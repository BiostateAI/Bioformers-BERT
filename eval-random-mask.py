import json
import argparse
from tqdm import tqdm

import torch
import evaluate
import numpy as np

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, get_scheduler

from sklearn.linear_model import LinearRegression

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
        self.vocab_size = num_genes
        self.hidden_size = model_hidden_size
        self.num_hidden_layers = model_hidden_layers
        self.num_attention_heads = model_attention_heads
        super().__init__()
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

def eval(config, model, device, test_dataset, test_dataloader):
    
    model.eval()
    # These are used to calculate overall prediction accuracy 
    total_masked_tokens = 0
    exact_match = 0
    off_by_one = 0
    # These are used to calculate prediction accuracy on genes with zero expression as ground truth
    total_zeros = 0
    zero_zero = 0
    zero_nonzero = 0
    # These are used to calculate prediction accuracy on genes with expression value 1 as ground truth
    total_ones = 0
    one_one = 0
    one_zero = 0
    one_other = 0
    # These are used to calculate prediction accuracy on genes with nonzero expression as ground truth
    total_nonzeros = 0
    nonzero_correct = 0
    nonzero_off_by_one = 0
    nonzero_off_by_two = 0
    
    total_highexp = 0
    highexp_correct = 0
    highexp_off_by_one = 0
    highexp_off_by_two = 0
    
    num_bins = test_dataset.num_bins
    
    for batch, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        
        x = x.to(device)
        y = y.to(device).view(-1)

        with torch.no_grad():
            outputs = model(x)
            outputs = outputs.view(-1, outputs.shape[-1])

        predictions = torch.argmax(torch.nn.functional.log_softmax(outputs, dim=-1), dim=-1)
        
        x_flat = x.view(-1)
        masked = (x_flat == test_dataset.mask_id)
        indices = torch.nonzero(masked)
        predictions = predictions.view(-1)[indices].view(-1)
        references = y.view(-1)[indices].view(-1)
        
        # print(predictions[:100])
        # print(references[:100])
        # print(predictions[:100] % num_bins)
        # print(references[:100] % num_bins)
        
        total_masked_tokens += len(predictions)
        exact_match += sum([predictions[i] == references[i] for i in range(predictions.shape[0])])
        off_by_one += sum([torch.abs(predictions[i] - references[i]) <= 1 for i in range(predictions.shape[0])])
        
        total_zeros += sum([references[i] % num_bins == 0 for i in range(predictions.shape[0])])
        zero_zero += sum([(predictions[i] % num_bins == 0 and references[i] % num_bins == 0) for i in range(predictions.shape[0])])
        zero_nonzero += sum([(predictions[i] % num_bins > 0 and references[i] % num_bins == 0) for i in range(predictions.shape[0])])
        
        total_ones += sum([references[i] % num_bins == 1 for i in range(predictions.shape[0])])
        one_one += sum([(predictions[i] % num_bins == 1 and references[i] % num_bins == 1) for i in range(predictions.shape[0])])
        one_zero += sum([(predictions[i] % num_bins == 0 and references[i] % num_bins == 1) for i in range(predictions.shape[0])])
        one_other += sum([(predictions[i] % num_bins > 1 and references[i] % num_bins == 1) for i in range(predictions.shape[0])])
        
        total_nonzeros += sum([references[i] % num_bins > 0 for i in range(predictions.shape[0])])
        nonzero_correct += sum([(references[i] % num_bins > 0 and predictions[i] == references[i]) for i in range(predictions.shape[0])])
        nonzero_off_by_one += sum([(references[i] % num_bins > 0 and torch.abs(predictions[i] - references[i]) <= 1) for i in range(predictions.shape[0])])
        nonzero_off_by_two += sum([(references[i] % num_bins > 0 and torch.abs(predictions[i] - references[i]) <= 2) for i in range(predictions.shape[0])])
        
        total_highexp += sum([references[i] % num_bins > 8 for i in range(predictions.shape[0])])
        highexp_correct += sum([(references[i] % num_bins > 8 and predictions[i] == references[i]) for i in range(predictions.shape[0])])
        highexp_off_by_one += sum([(references[i] % num_bins > 8 and torch.abs(predictions[i] - references[i]) <= 1) for i in range(predictions.shape[0])])
        highexp_off_by_two += sum([(references[i] % num_bins > 8 and torch.abs(predictions[i] - references[i]) <= 2) for i in range(predictions.shape[0])])
        
        total_highexp += sum([references[i] % num_bins > config['bin_cutoff'] for i in range(predictions.shape[0])])
        highexp_correct += sum([(references[i] % num_bins > config['bin_cutoff'] and predictions[i] == references[i]) for i in range(predictions.shape[0])])
        highexp_off_by_one += sum([(references[i] % num_bins > config['bin_cutoff'] and torch.abs(predictions[i] - references[i]) <= 1) for i in range(predictions.shape[0])])
        highexp_off_by_two += sum([(references[i] % num_bins > config['bin_cutoff'] and torch.abs(predictions[i] - references[i]) <= 2) for i in range(predictions.shape[0])])
        
    print("Total masked tokens:", total_masked_tokens)
    print("Exact match sensitivity:", exact_match/total_masked_tokens)
    print("Off by one sensitivity:", off_by_one/total_masked_tokens)
    
    print("Total token with ground truth of 0:", int(total_zeros))
    print("Zero tokens predicted correctly:", zero_zero/total_zeros)
    print("Zero tokens predicted incorrectly:", zero_nonzero/total_zeros)
    
    print("Total token with ground truth of 1:", int(total_ones))
    print("One tokens predicted correctly:", one_one/total_ones)
    print("One tokens predicted to be zero:", one_zero/total_ones)
    print("One tokens predicted to other values:", one_other/total_ones)
    
    print("Total token with ground truth of nonzero values:", int(total_nonzeros))
    print("Nonzero tokens predicted correctly:", nonzero_correct/total_nonzeros)
    print("Nonzero tokens predicted off by one:", nonzero_off_by_one/total_nonzeros)
    print("Nonzero tokens predicted off by two:", nonzero_off_by_two/total_nonzeros)
    
    print("Total token with ground truth greater than bin_cutoff:", int(total_highexp))
    print("HEL tokens predicted correctly:", highexp_correct/total_highexp)
    print("HEL tokens predicted off by one:", highexp_off_by_one/total_highexp)
    print("HEL tokens predicted off by two:", highexp_off_by_two/total_highexp)
    
    
def main(checkpoint_path):
    
    with open("./settings.json") as f:
        config = json.loads(f.read())
    
    train_data = np.load("./processed_data/train_data.npy")
    test_data = np.load("./processed_data/valid_data.npy")
    
    num_bins = np.max([np.max(train_data), np.max(test_data)]) + 1
    num_genes = train_data.shape[1]
    mask_id = num_bins * num_genes
    
    for i in range(num_genes):
        train_data[:, i] += num_bins * i
        test_data[:, i] += num_bins * i
        
    test_dataset = GeneDataset(test_data, mask_id, num_bins, config["mask_ratio"], config["nonzero_ratio"])

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

    model.load_state_dict(torch.load(checkpoint_path))

    eval(config, model, device, test_dataset, test_dataloader)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="path to model checkpoint")
    args = parser.parse_args()
    main(args.checkpoint)