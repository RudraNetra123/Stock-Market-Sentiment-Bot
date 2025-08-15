import pandas as pd
import torch
from torch.utils.data import Dataset

class SentimentPriceDataset(Dataset):
    def __init__(self, csv_file, seq_len=10):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len

        # normalize if needed
        self.data['delta_score'] = self.data['delta_score']
        self.data['next_change'] = self.data['next_change']

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq_delta = self.data['delta_score'].iloc[idx : idx + self.seq_len].values
        target = self.data['next_change'].iloc[idx + self.seq_len]

        seq_delta = torch.tensor(seq_delta, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return seq_delta.unsqueeze(-1), target
