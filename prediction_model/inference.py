import os
import json
import random
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
from accelerate import Accelerator


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048, prompt_style=0, is_inference=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_style = prompt_style
        self.is_inference = is_inference
        self.has_label = 'd_5' in data.columns
        if not is_inference and not self.has_label:
             print("Warning: Training/Evaluation mode expected, but 'd_5' column is missing in data.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]     
        if self.has_label:
            label = float(row['d_5'])
        else:
            label = 0.0 
        if self.prompt_style == 0 or self.prompt_style == 1: 
            text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its disruption:'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        else:
            raise NotImplementedError('prompt_style not registed in dataset.')

seed = 42
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_args():
    p = argparse.ArgumentParser(description="Inference script for disruption regression model.")
    p.add_argument('--data_path', type=str, required=True, help="Path to the input JSONL file with 'title' and 'abstract'.")
    p.add_argument('--weight_dir', type=str, required=True, help="Path to the fine-tuned model weights directory.")
    p.add_argument('--output_dir', type=str, default='inference_results', help="Directory to save the prediction results.")
    p.add_argument('--max_length', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_labels', type=int, default=1, help="Should be 1 for regression.")
    p.add_argument('--load_in_8bit', action='store_true')
    
    args = p.parse_args()
    return args

def main():
    args = get_args()
    accelerator = Accelerator()
    
    model_name = os.path.basename(args.weight_dir.strip('/'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(args.output_dir, f"{model_name}_inference_{timestamp}")

    if accelerator.is_main_process:
        os.makedirs(runs_dir, exist_ok=True)
        args.results_file = os.path.join(runs_dir, "predictions.csv")
        args.metrics_file = os.path.join(runs_dir, "metrics.json")
        with open(os.path.join(runs_dir, 'inference_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.weight_dir)
    tokenizer.pad_token = tokenizer.eos_token
    device_map = {'': accelerator.device}
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        args.weight_dir, num_labels=args.num_labels, load_in_8bit=args.load_in_8bit,
        device_map=device_map, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    accelerator.print(f"Model loaded from {args.weight_dir}")

    try:
        input_df = pd.read_json(args.data_path)
    except Exception:
        try:
            input_df = pd.read_csv(args.data_path)
        except Exception as e:
            accelerator.print(f"Error loading data from {args.data_path}: {e}")
            return
    
    if not all(col in input_df.columns for col in ['title', 'abstract']):
        accelerator.print(f"Error: Input file must contain 'title' and 'abstract' columns.")
        return

    dataset = TextDataset(input_df, tokenizer, max_length=args.max_length, is_inference=True) 
    inference_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    accelerator.print(f'Inference Dataloader has {len(dataset)} samples.')

    all_pred = []
    model, inference_loader = accelerator.prepare(model, inference_loader)

    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            predictions = outputs.logits.squeeze(-1)
            all_pred.append(predictions)

    all_pred = accelerator.gather_for_metrics(torch.cat(all_pred, dim=0))

    if accelerator.is_main_process:
        all_pred_np = all_pred.float().cpu().numpy()

        df_results = input_df.iloc[:len(all_pred_np)].copy()
        df_results['prediction'] = all_pred_np
        】
        prediction_mean = float(df_results['prediction'].mean())
        】
        output_cols = ['prediction'] + [col for col in input_df.columns]
        output_cols = list(dict.fromkeys(output_cols))
        df_results[output_cols].to_csv(args.results_file, index=False)
        】
        metrics = {
            "prediction_mean": prediction_mean,
            "total_samples": len(df_results)
        }
        with open(args.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nPrediction results saved to: {args.results_file}")
        print(f"Prediction mean: {prediction_mean:.4f}")
        print(f"Total samples: {len(df_results)}")
        print(f"Results saved to directory: {runs_dir}")


if __name__ == "__main__":
    main()