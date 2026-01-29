import os
import json
import random
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
from accelerate import Accelerator

from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
from scipy.stats import spearmanr, pearsonr, kendalltau
import matplotlib.pyplot as plt

from train import TextDataset

seed = 42
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def rank_percentile(x: np.ndarray) -> np.ndarray:
    """将连续值映射到[0,100]分位。ties使用平均秩。"""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x)+1)
    pct = (ranks - 0.5) / len(x) * 100.0
    return pct

def generate_report(df_results, args):
    all_pred_np = df_results['prediction'].to_numpy()
    all_gt_np   = df_results['ground_truth'].to_numpy()

    rho = spearmanr(all_gt_np, all_pred_np).correlation

    y_bin_gt0 = (all_gt_np > args.auc_threshold).astype(int)
    auc_gt0 = roc_auc_score(y_bin_gt0, all_pred_np)

    q_top5 = np.quantile(all_gt_np, 0.95) 
    y_bin_top5 = (all_gt_np >= q_top5).astype(int)
    auc_top5 = roc_auc_score(y_bin_top5, all_pred_np)

    report = []
    report.append(f"Model Path: {args.weight_dir}")
    report.append(f"Test Data Path: {args.data_path}")
    report.append(f"Total Samples: {len(df_results)}")
    report.append(f"Spearman Correlation: {rho:.4f}")
    report.append(f"AUC (GT > {args.auc_threshold}): {auc_gt0:.4f}")
    report.append(f"AUC (Top-5%): {auc_top5:.4f}")
    
    report_text = "\n".join(report)
    print(report_text)
    metrics = {
        "spearman": float(rho),
        "auc_gt>0": float(auc_gt0),
        "auc_top5": float(auc_top5)
    }
    with open(os.path.join(args.runs_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def get_args():
    p = argparse.ArgumentParser(description="Extended evaluation for disruption regression model.")
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--weight_dir', type=str, required=True)
    p.add_argument('--runs_dir', type=str, default=None)
    p.add_argument('--auc_threshold', type=float, default=0.0)
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_labels', type=int, default=1)
    p.add_argument('--load_in_8bit', action='store_true')
    p.add_argument('--prompt_style', type=int, default=0)
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    if args.runs_dir is None:
        model_name = os.path.basename(args.weight_dir.strip('/'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.runs_dir = os.path.join('evaluation_results', f"{model_name}_{timestamp}")
    return args

def main():
    args = get_args()
    accelerator = Accelerator()

    if accelerator.is_main_process:
        os.makedirs(args.runs_dir, exist_ok=True)
        with open(os.path.join(args.runs_dir, 'evaluation_args.json'), 'w') as f:
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
    print(model)

    full_data_df = pd.read_json(args.data_path, lines=True)
    dataset = TextDataset(full_data_df, tokenizer, max_length=args.max_length,prompt_style=args.prompt_style)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    accelerator.print(f'Test Dataloader has {len(dataset)} samples.')

    all_pred, all_gt = [], []
    model, test_loader = accelerator.prepare(model, test_loader)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Inference", disable=not accelerator.is_local_main_process):
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            predictions = outputs.logits.squeeze(-1)
            labels = batch["labels"]
            all_gt.append(labels); all_pred.append(predictions)

    all_pred = accelerator.gather_for_metrics(torch.cat(all_pred, dim=0))
    all_gt   = accelerator.gather_for_metrics(torch.cat(all_gt,   dim=0))

    if accelerator.is_main_process:
        all_pred_np = all_pred.float().cpu().numpy()
        all_gt_np   = all_gt.float().cpu().numpy()

        df_results = full_data_df.iloc[:len(all_gt_np)].copy()
        df_results['ground_truth'] = all_gt_np
        df_results['prediction']   = all_pred_np

        df_tmp = df_results.copy()
        df_tmp['gt_pct']   = rank_percentile(df_tmp['ground_truth'].to_numpy())
        df_tmp['pred_pct'] = rank_percentile(df_tmp['prediction'].to_numpy())

        results_csv = os.path.join(args.runs_dir, "predictions_review.csv")
        generated_cols = ['ground_truth', 'prediction', 'gt_pct', 'pred_pct']
        original_cols = list(full_data_df.columns)
        cols_to_save = generated_cols + [col for col in original_cols if col not in generated_cols]
        df_tmp[cols_to_save].to_csv(results_csv, index=False)

        generate_report(df_results, args)


if __name__ == "__main__":
    main()
