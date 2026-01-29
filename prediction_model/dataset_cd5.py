
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048, prompt_style=0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_style = prompt_style

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = float(row['d_5'])
        if self.prompt_style == 0:
            text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its disruption:'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        else:
            raise NotImplementedError('prompt_style not registed in NAID/dataset.py')