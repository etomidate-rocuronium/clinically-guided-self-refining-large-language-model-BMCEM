import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

                           
        
                           
DATA_DIR = "intermediate"
OUTPUT_DIR = "results"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

ALL_SEEDS = [7, 42, 99, 123, 2024]
NUM_FOLDS = 5
MAX_LEN = 512
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

                           
         
                           
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

                           
         
                           
def compute_metrics(y_true, y_probs):
    y_pred = (y_probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": roc_auc_score(y_true, y_probs),
        "auprc": average_precision_score(y_true, y_probs),
        "brier": brier_score_loss(y_true, y_probs),
    }

                           
                     
                           
def mean_std(values):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
                                            
    return mean, std

                           
                
                           
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

                           
           
                           
all_results = []

for seed in ALL_SEEDS:
    for fold in range(NUM_FOLDS):

        print(f"\n🚀 Running Seed {seed} | Fold {fold}")

        set_seed(seed)

        train_path = os.path.join(DATA_DIR, f"train_seed{seed}_fold{fold}.pkl")
        val_path = os.path.join(DATA_DIR, f"val_seed{seed}_fold{fold}.pkl")

        df_train = pd.read_pickle(train_path)
        df_val = pd.read_pickle(val_path)

                                   
                                              
                                   
        df_train = df_train[['translated_summary_cleaned', 'y']]
        df_val = df_val[['translated_summary_cleaned', 'y']]

        df_train['y'] = df_train['y'].map({1: 'yes', 0: 'no'})
        df_val['y'] = df_val['y'].map({1: 'yes', 0: 'no'})

        label_map = {'no': 0, 'yes': 1}
        y_train = df_train['y'].map(label_map).values
        y_val = df_val['y'].map(label_map).values

        X_train = df_train['translated_summary_cleaned'].tolist()
        X_val = df_val['translated_summary_cleaned'].tolist()

                                   
                  
                                   
        train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
        val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LEN)

                                   
               
                                   
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )
        model.to(DEVICE)

                                   
                  
                                   
        training_args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, f"model_seed{seed}_fold{fold}"),
            learning_rate=LR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            logging_steps=50,
            save_strategy="no",
            eval_strategy="no",
            seed=seed,
            fp16=torch.cuda.is_available()
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

                                   
                    
                                   
        preds = trainer.predict(val_dataset)
        logits = preds.predictions

        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].cpu().numpy()

                                   
                 
                                   
        metrics = compute_metrics(y_val, probs)

        print(" print metrics:", metrics)

                                   
                     
                                   
        output_data = {
            "seed": seed,
            "fold": fold,
            "y_true": y_val,
            "y_prob": probs,
            "metrics": metrics
        }

        save_path = os.path.join(OUTPUT_DIR, f"pred_seed{seed}_fold{fold}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(output_data, f)

        all_results.append(metrics)

                           
                   
                           
print("\n============================")
print(" final results (mean ± std)")                                                                                                                                                                        
print("============================\n")

metrics_keys = all_results[0].keys()

final_summary = {}

for key in metrics_keys:
    values = [r[key] for r in all_results]
    mean, std = mean_ci(values)
    final_summary[key] = {"mean": mean, "std": std}

    print(f"{key}: {mean:.4f} ± {std:.4f}")

        
with open(os.path.join(OUTPUT_DIR, "bioclinical_bert_all_metrics.pkl"), "wb") as f:
    pickle.dump(final_summary, f)

print("\n Hooray!")
