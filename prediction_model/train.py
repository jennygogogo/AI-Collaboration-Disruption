from transformers import Trainer, TrainingArguments
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, FlaxLlamaForCausalLM
import pandas as pd
from torch.utils.data.dataset import random_split
import argparse
import json
from accelerate import Accelerator
import os
import torch.nn as nn

from dataset_cd5 import TextDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).squeeze()
    labels = torch.tensor(labels).squeeze()
    mse = nn.MSELoss()(predictions, labels).item()
    mae = nn.L1Loss()(predictions, labels).item()
    predictions_np = predictions.detach().numpy()
    labels_np = labels.detach().numpy()
    spearman_corr, _ = spearmanr(predictions_np, labels_np)
    pearson_corr, _ = pearsonr(predictions_np, labels_np)
    r2 = r2_score(labels_np, predictions_np)
    return {"mse": mse, "mae": mae,"spearman": spearman_corr,"pearson": float(pearson_corr),"r2": float(r2)}



def save_args_to_json(args, file_path):
    args_dict = vars(args)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    df_train = pd.read_json(args.data_path, lines=True)
    df_val = pd.read_json(args.val_data_path, lines=True) if args.val_data_path else None
    df_test = pd.read_json(args.test_data_path, lines=True) if args.test_data_path else None

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=args.num_labels,
        load_in_8bit=args.load_in_8bit,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    target_modules = [m.strip() for m in args.target_modules.split(',') if m.strip()]
    if not target_modules:
        for param in model.model.parameters():
            param.requires_grad = False
  
    if len(target_modules) > 0:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False
        )
        if args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        

    train_dataset = TextDataset(df_train, tokenizer, args.max_length,args.prompt_style)
    eval_dataset = TextDataset(df_val, tokenizer, args.max_length,args.prompt_style)

    if args.runs_dir is None:
        args.runs_dir = os.path.join(
            'runs_qwen3_seqcls', datetime.now().strftime("%m-%d-%H-%M-%S")
        )
    os.makedirs(args.runs_dir, exist_ok=True)
    save_args_to_json(args, os.path.join(args.runs_dir, "args.json"))

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,
        output_dir=args.runs_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.total_epochs,
        logging_dir=args.runs_dir,
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(os.path.join(args.runs_dir, "last"))

def get_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-0.6B with LoRA for binary text classification (no thinking/chat)."
    )

    parser.add_argument('--checkpoint', type=str, default='llama3_weight', help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--data_path', type=str, default='NAID/NAID_test_extrainfo_arxiv_id.csv',
                        help='Path to the training dataset CSV file')
    parser.add_argument('--test_data_path', type=str, default='NAID/NAID_train_extrainfo_arxiv_id.csv',
                        help='Path to the testing dataset CSV file')
    parser.add_argument('--val_data_path', type=str, default='NAID/NAID_train_extrainfo_arxiv_id.csv',
                        help='Path to the testing dataset CSV file')
    parser.add_argument('--runs_dir', type=str, default=None,
                        help='Directory for storing TensorBoard logs and model checkpoints')

    parser.add_argument('--total_epochs', type=int, default=5, help='Total number of epochs to train')
    parser.add_argument('--base_lr', type=float, default=5e-5, help='Base learning rate for the optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for the optimizer')

    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of the tokenized input sequences')
    parser.add_argument('--loss_func', type=str, default='mse', choices=['bce', 'mse', 'l1', 'smoothl1', 'focalmse'],
                        help='Loss function to use')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for sequence classification')
    parser.add_argument('--load_in_8bit', type=bool, default=False,
                        help='Whether to load the model in 8-bit for efficiency')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on (cuda or cpu)')
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Expansion factor for LoRA layers')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout rate for LoRA layers')
    parser.add_argument('--target_modules', type=str, default='q_proj,v_proj',
                        help='Comma-separated list of transformer modules to apply LoRA')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--prompt_style', type=int, default=0)  

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
