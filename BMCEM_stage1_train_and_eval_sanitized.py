base_dir_model = "<MODEL_OUTPUT_DIR>"
base_dir_results = "<RESULTS_OUTPUT_DIR>"

import os
import gc
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.core.numeric
import sys

sys.modules['numpy._core.numeric'] = numpy.core.numeric


import torch
import torch.distributed as dist

from tqdm import tqdm
from datasets import Dataset
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve, precision_recall_curve
)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

dist.init_process_group(backend="nccl", init_method="env://")

device = torch.device(f"cuda:{local_rank}")
rank = dist.get_rank()

ALL_SEEDS = [7, 42, 99, 123, 2024]
N_SPLITS = 5

base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
<HF_TOKEN> = "<HF_TOKEN>"

SYSTEM_PROMPT = (
    "You are a neurologist working in the emergency department. "
    "Your task is to read the following patient summary and decide "
    "if the patient is likely to be diagnosed acute stroke or not. "
    "Please respond ONLY with the word 'yes' or 'no'."
)

def generate_prompt(data_point):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"patient summary:\n{data_point['translated_summary_cleaned']}"
        },
        {
            "role": "assistant",
            "content": data_point["y"]
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def generate_test_prompt(data_point):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"patient summary:\n{data_point['translated_summary_cleaned']}"
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=<HF_TOKEN>)
tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

import bitsandbytes as bnb
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, cls):
            names.add(n.split('.')[-1])
    names.discard("lm_head")
    return list(names)

def predict_proba(df, model, tokenizer):
    model.eval()

    y_pred = []
    y_prob = []

    yes_token = tokenizer(" yes", add_special_tokens=False).input_ids[0]
    no_token  = tokenizer(" no",  add_special_tokens=False).input_ids[0]

    for i in tqdm(range(len(df)), disable=(rank != 0)):

        inputs = tokenizer(df.iloc[i]["text"], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[:, -1, :]
        binary_logits = logits[0, [no_token, yes_token]] / 0.1
        binary_probs = torch.softmax(binary_logits, dim=0)
        p = binary_probs[1].item()

        y_prob.append(p)
        y_pred.append("yes" if p >= 0.5 else "no")

    return y_pred, np.array(y_prob)

if rank == 0:
    all_metrics = []
    all_fpr, all_tpr = [], []
    all_precision, all_recall = [], []

    all_y_true = []
    all_y_prob = []

for seed in ALL_SEEDS:
    set_seed(seed)

    for fold in range(N_SPLITS):

        if rank == 0:
            print(f"\n Seed {seed} | Fold {fold}")

        df_train = pd.read_pickle("<TRAIN_OR_VAL_PICKLE>")
        df_val   = pd.read_pickle("<TRAIN_OR_VAL_PICKLE>")

        df_val = df_val[['translated_summary_cleaned', 'y']]
        df_val['y'] = df_val['y'].map({1: 'yes', 0: 'no'})

        df_train = df_train[['translated_summary_cleaned', 'y']]
        df_train['y'] = df_train['y'].map({1: 'yes', 0: 'no'})

        df_train["text"] = df_train.apply(generate_prompt, axis=1)
        df_val["text"]   = df_val.apply(generate_prompt, axis=1)

        y_true = (df_val["y"] == "yes").astype(int).values

        X_test = pd.DataFrame(
            df_val.apply(generate_test_prompt, axis=1),
            columns=["text"]
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": device},
            token=<HF_TOKEN>,
        )

        model.use_cache = False

        peft_config = LoraConfig(
            r=32,
            lora_alpha=1,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=find_all_linear_names(model),
        )

        model = get_peft_model(model, peft_config)

        train_data = Dataset.from_pandas(df_train[["text"]])
        eval_data  = Dataset.from_pandas(df_val[["text"]])

        output_dir = os.path.join(base_dir_model, f"seed{seed}_fold{fold}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            optim="paged_adamw_32bit",
            learning_rate=2e-3,
            weight_decay=0.01,
            fp16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.3,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=10,
            logging_steps=10,
            save_strategy="no",
            report_to="tensorboard",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=peft_config,
        )

        trainer.train()

        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        dist.barrier()

        y_pred, y_prob = predict_proba(X_test, model, tokenizer)

        if rank == 0:

            f1 = f1_score(y_true, (np.array(y_pred)=="yes").astype(int))
            acc = accuracy_score(y_true, (np.array(y_pred)=="yes").astype(int))

            tn, fp, fn, tp = confusion_matrix(y_true, (np.array(y_pred)=="yes").astype(int)).ravel()

            sensitivity = tp/(tp+fn)
            specificity = tn/(tn+fp)

            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)

            all_metrics.append({
                "f1": f1,
                "accuracy": acc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "auroc": auroc,
                "auprc": auprc,
                "brier": brier,
            })

            all_y_true.append(y_true)
            all_y_prob.append(y_prob)

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            all_fpr.append(fpr)
            all_tpr.append(tpr)

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            all_precision.append(precision)
            all_recall.append(recall)

            print(
                f"F1: {f1:.4f} | "
                f"ACC: {acc:.4f} | "
                f"SENS: {sensitivity:.4f} | "
                f"SPEC: {specificity:.4f} | "
                f"AUROC: {auroc:.4f} | "
                f"AUPRC: {auprc:.4f} | "
                f"Brier: {brier:.4f}"
            )

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

        dist.barrier()

if rank == 0:

    def compute_ci(x):
        x = np.array(x)
        m = x.mean()
        ci = 1.96 * x.std(ddof=1)/np.sqrt(len(x))
        return m, m-ci, m+ci

    print("\n final results ")
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        seed_means = [np.mean(vals[i:i + N_SPLITS]) for i in range(0, len(vals), N_SPLITS)]
        m,l,u = compute_ci(seed_means)
        print(f"{k}: {m:.4f} ({l:.4f}-{u:.4f})")
    
    mean_fpr = np.linspace(0,1,100)
    tprs = [np.interp(mean_fpr, f, t) for f,t in zip(all_fpr, all_tpr)]
    mean_tpr = np.mean(tprs, axis=0)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr)
    plt.savefig(os.path.join(base_dir_results,"roc.png"))
    plt.close()

    mean_recall = np.linspace(0,1,100)
    precs = [np.interp(mean_recall, r[::-1], p[::-1]) for p,r in zip(all_precision, all_recall)]
    mean_prec = np.mean(precs, axis=0)

    plt.figure()
    plt.plot(mean_recall, mean_prec)
    plt.savefig(os.path.join(base_dir_results,"pr.png"))
    plt.close()

    with open(os.path.join(base_dir_results, "stage1_final_results.txt"), "w") as f:
        f.write("===== FINAL RESULTS =====\n")
        for k in all_metrics[0]:
            vals = [m[k] for m in all_metrics]
            seed_means = [np.mean(vals[i:i + N_SPLITS]) for i in range(0, len(vals), N_SPLITS)]
            m, l, u = compute_ci(seed_means)
            f.write(f"{k}: {m:.4f} ({l:.4f}-{u:.4f})\n")
        
    with open(os.path.join(base_dir_results, "stage1_roc_data.pkl"), "wb") as f:
        pickle.dump({
            "all_fpr": all_fpr,
            "all_tpr": all_tpr
        }, f)
    
    with open(os.path.join(base_dir_results, "stage1_pr_data.pkl"), "wb") as f:
        pickle.dump({
            "all_precision": all_precision,
            "all_recall": all_recall
        }, f)
    
    with open(os.path.join(base_dir_results, "stage1_metrics_all.pkl"), "wb") as f:
        pickle.dump(all_metrics, f)

    with open(os.path.join(base_dir_results, "stage1_predictions_all.pkl"), "wb") as f:
        pickle.dump({
            "y_true": all_y_true,
            "y_prob": all_y_prob
        }, f)
    
    print("\n hooray!")

dist.destroy_process_group()










