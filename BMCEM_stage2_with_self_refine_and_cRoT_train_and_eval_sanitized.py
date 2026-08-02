base_dir_stage1 = "<STAGE1_MODEL_DIR>"
base_dir_stage2 = "<STAGE2_OUTPUT_DIR>"

from sklearn.metrics import(
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from tqdm import tqdm
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

import bitsandbytes as bnb
import numpy.core.numeric
import sys

sys.modules['numpy._core.numeric'] = numpy.core.numeric

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
device = torch.device(f"cuda:{local_rank}")
rank = dist.get_rank()

ALL_SEEDS = [7, 42, 99, 123, 2024]
N_SPLITS = 5

base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
<HF_TOKEN> = "<HF_TOKEN>"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=<HF_TOKEN>)
tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, cls):
            names.add(n.split('.')[-1])
    names.discard("lm_head")
    return list(names)

SYSTEM_PROMPT = (
    "You are a neurologist working in the emergency department. "
    "Your task is to read the following patient summary and decide "
    "if the patient is likely to be diagnosed acute stroke or not. "
    "Please respond ONLY with the word 'yes' or 'no'."
)

def stage1_prompt(row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"patient summary:\n{row['translated_summary_cleaned']}"
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

SYSTEM_PROMPT_STAGE2 = (
    "You are a neurologist working in the emergency department."
)

def stage2_user_content(row):
    return f"""
Given the patient summary below and the previous prediction '{row["pred_stage1"]}', please reconsider and provide the correct diagnosis ('yes' or 'no').

Patient summary:
{row["translated_summary_cleaned"]}

Previous prediction:
{row["pred_stage1"]}

Note the following rule of thumb.
1. The more the number of neurological deficits present, the likelier of true stroke.
2. Presence of hypertension, diabetes mellitus, dyslipidemia increases the likelihood of true stroke.
3. Presence of cancer increases the likelihood of true stroke.
4. Presence of unilateral side weakness increases the likelihood of true stroke.
5. History of epilepsy or seizure decreases the likelihood of true stroke.
6. Presence of symmetric weakness decreases the likelihood of true stroke.
""".strip()

def stage2_train_prompt(row):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_STAGE2,
        },
        {
            "role": "user",
            "content": stage2_user_content(row),
        },
        {
            "role": "assistant",
            "content": row["y"],
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def stage2_test_prompt(row):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_STAGE2,
        },
        {
            "role": "user",
            "content": stage2_user_content(row),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def load_stage1(path, is_trainable=False):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": device},
        token=<HF_TOKEN>,
    )
    model = PeftModel.from_pretrained(base, path, is_trainable=is_trainable)
    model.eval()
    return model

def generate_preds(df, model):
    preds = []
    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_id  = tokenizer("no", add_special_tokens=False).input_ids[0]

    for i in tqdm(range(len(df)), disable=(rank!=0)):
        inputs = tokenizer(
            stage1_prompt(df.iloc[i]),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)

        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        pred = "yes" if probs[0, yes_id] >= probs[0, no_id] else "no"
        preds.append(pred)

    return preds

def evaluate_model(df, model):
    model.eval()

    y_true = df["y"].tolist()

    yes_ids = tokenizer("yes", add_special_tokens=False).input_ids
    no_ids = tokenizer("no", add_special_tokens=False).input_ids

    assert len(yes_ids) == 1, f"'yes' is not one token: {yes_ids}"
    assert len(no_ids) == 1, f"'no' is not one token: {no_ids}"

    yes_id = yes_ids[0]
    no_id = no_ids[0]

    y_pred = []
    y_prob = []

    for i in tqdm(range(len(df)), disable=(rank!=0)):
        inputs = tokenizer(
            stage2_test_prompt(df.iloc[i]),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)

        logits = out.logits[:, -1, :]
        binary_logits = logits[0, [no_id, yes_id]] / 0.1
        binary_probs = torch.softmax(binary_logits, dim=0)

        prob_yes = binary_probs[1].item()
        pred = "yes" if prob_yes >= 0.5 else "no"

        y_pred.append(pred)
        y_prob.append(prob_yes)

    y_true_bin = [1 if y=="yes" else 0 for y in y_true]
    y_pred_bin = [1 if y=="yes" else 0 for y in y_pred]

    acc = accuracy_score(y_true, y_pred)
    f1_yes = f1_score(y_true, y_pred, pos_label="yes")

    precision = precision_score(y_true, y_pred, pos_label="yes")
    sensitivity = recall_score(y_true, y_pred, pos_label="yes")

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
    specificity = tn / (tn + fp + 1e-8)

    auroc = roc_auc_score(y_true_bin, y_prob)
    auprc = average_precision_score(y_true_bin, y_prob)
    brier = brier_score_loss(y_true_bin, y_prob)

    return {
        "acc": acc,
        "f1_yes": f1_yes,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier
    }

all_metrics = []

for seed in ALL_SEEDS:
    set_seed(seed)

    for fold in range(N_SPLITS):

        if rank==0:
            print(f"\n Stage2 Train | Seed {seed} Fold {fold}")

        df = pd.read_pickle("<DATA_SPLIT_PICKLE_PATH>")
        df = df[['translated_summary_cleaned','y']]
        df['y'] = df['y'].map({1:'yes',0:'no'})

        stage1 = load_stage1(
            os.path.join(base_dir_stage1,f"seed{seed}_fold{fold}"),
            is_trainable=True,
        )
        df["pred_stage1"] = generate_preds(df, stage1)

        df = df.reset_index(drop=True)

        df["text"] = df.apply(stage2_train_prompt, axis=1)
        dataset = Dataset.from_pandas(df[["text"]])

        model = stage1
        model.use_cache = False

        output_dir = "<OUTPUT_DIR>"

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            optim="paged_adamw_32bit",
            learning_rate=2e-3,
            weight_decay=0.01,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            warmup_ratio=0.3,
            group_by_length=False,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="no",
            report_to="tensorboard",
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
        )

        trainer.train()

        val_df = pd.read_pickle("<DATA_SPLIT_PICKLE_PATH>")
        val_df = val_df[['translated_summary_cleaned','y']]
        val_df['y'] = val_df['y'].map({1:'yes',0:'no'})

        stage1 = load_stage1(os.path.join(base_dir_stage1,f"seed{seed}_fold{fold}"))
        val_df["pred_stage1"] = generate_preds(val_df, stage1)

        del stage1
        torch.cuda.empty_cache()
        gc.collect()

        metrics = evaluate_model(val_df, model)

        if rank == 0:
            print(f"\n Seed {seed} Fold {fold} Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

        all_metrics.append(metrics)

        out = os.path.join(base_dir_stage2,f"seed{seed}_fold{fold}")
        if rank==0:
            os.makedirs(out, exist_ok=True)
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)

        dist.barrier()

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

dist.destroy_process_group()
print(" hooray Stage2 training finished")
