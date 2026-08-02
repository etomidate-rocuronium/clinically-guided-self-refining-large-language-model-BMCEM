import os
import time
import pickle
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank >= 0:
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
base_dir_model = "<MODEL_DIR>"
base_dir_results = "<RESULTS_DIR>"
check_point = "<CHECKPOINT_DIR>"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map={"": device},
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

df = pd.read_pickle("<DATASET_PATH>")
df = df[["summary_en", "y"]]
df["y"] = df["y"].map({1: "yes", 0: "no"})

df = (
    df.groupby('y', group_keys=False)
      .apply(lambda x: x.sample(n=1000, random_state=42))
      .reset_index(drop=True)
)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df['y'].value_counts())

def generate_prompt(data_point):
    return f"""
            You are a neurologist working in the emergency department. Your task is to read the following patient summary and decide if the patient is likely to be diagnosed acute stroke or not. Please respond ONLY with the word 'yes' or 'no'.
            Patient summary: {data_point["summary_en"]}
            label: {data_point["y"]}""".strip()

def generate_test_prompt(data_point):
    return f"""
            You are a neurologist working in the emergency department. Your task is to read the following patient summary and decide if the patient is likely to be diagnosed acute stroke or not. Please respond ONLY with the word 'yes' or 'no'.
            Patient summary: {data_point["summary_en"]}
            label: """.strip()

def build_reconsider_prompt(summary: str, pred: str, true: str = "", reveal_true: bool = False) -> str:
    true_part = true if reveal_true else ""
    return f"""
            You are a neurologist working in the emergency department.
            Given the patient summary below and the previous prediction '{pred}', please reconsider and provide the correct diagnosis ('yes' or 'no').

            Patient summary: {summary}
            Previous prediction: {pred}

            Note the following rule of thumb.
            1. The more the number of neurological deficits present, the likelier of true stroke.
            2. Presence of hypertension, diabetes mellitus, dyslipidemia increases the likelihood of true stroke.
            3. Presence of cancer increases the likelihood of true stroke.
            4. Presence of unilateral side weakness increases the likelihood of true stroke.
            5. History of epilepsy or seizure decreases the likelihood of true stroke.
            6. Presence of symmetric weakness decreases the likelihood of true stroke.

            label: {true_part}""".strip()

def align_attention_to_words(tokens, scores):
    words = []
    word_scores = []
    current_word = ""
    current_scores = []
    for token, score in zip(tokens, scores):
        if token.startswith("Ġ"):
            if current_word:
                words.append(current_word)
                word_scores.append(np.mean(current_scores))
            current_word = token[1:]
            current_scores = [score]
        else:
            current_word += token
            current_scores.append(score)
    if current_word:
        words.append(current_word)
        word_scores.append(np.mean(current_scores))
    return words, word_scores

def extract_yes_no(generated_text: str, categories=("yes", "no")) -> str:

    if "label:" in generated_text:
        answer = generated_text.split("label:")[-1].strip()
        for cat in categories:
            if cat.lower() in answer.lower():
                return cat

    tail = generated_text[-32:].lower()
    for cat in categories:
        if cat in tail:
            return cat
    return "none"

def predict_with_attention_rollout(test_df, model, tokenizer, device, visualize=False):
    y_pred = []
    attention_outputs = []
    categories = ["yes", "no"]

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_df))):
            prompt = test_df.iloc[i]["text"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model(**inputs, output_attentions=True)

            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            predicted_token_id = torch.argmax(next_token_logits, dim=-1)

            full_token_ids = torch.cat([inputs["input_ids"], predicted_token_id.unsqueeze(-1)], dim=-1)[0]
            generated_text = tokenizer.decode(full_token_ids, skip_special_tokens=True)

            pred = extract_yes_no(generated_text, categories=categories)
            y_pred.append(pred)

            layer_attentions = []
            for layer_att in outputs.attentions:
                avg_layer_att = layer_att.mean(dim=1)
                layer_attentions.append(avg_layer_att)

            max_seq_len = max(att.shape[-1] for att in layer_attentions)
            padded_matrices = []
            for att in layer_attentions:
                seq_len = att.shape[-1]
                identity = torch.eye(seq_len, device=device).unsqueeze(0)
                mat = (att + identity) / 2.0
                if seq_len < max_seq_len:
                    pad_amt = max_seq_len - seq_len
                    mat = F.pad(mat, (0, pad_amt, 0, pad_amt), value=0)
                padded_matrices.append(mat)

            rollout = padded_matrices[0]
            for mat in padded_matrices[1:]:
                rollout = torch.matmul(rollout, mat)
            rollout = rollout / rollout.sum(dim=-1, keepdim=True)

            last_token_idx = rollout.shape[-1] - 1
            token_scores = rollout[0, last_token_idx, :]
            token_scores = token_scores / token_scores.sum()

            tokens = tokenizer.convert_ids_to_tokens(full_token_ids)
            attention_outputs.append(list(zip(tokens, token_scores.cpu().tolist())))

            detokenized_text = tokenizer.convert_tokens_to_string(tokens)
            print(detokenized_text)
            print(tokens)

    return y_pred, attention_outputs

start_time = time.time()

seeds = [7, 42, 99, 123, 2024]
n_splits = 5

reveal_true_in_prompt = False

for seed in seeds:
    print(f"\n Seed: {seed}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["y"])):
        print(f"\n Fold {fold + 1}/{n_splits}")

        stage1_dir = os.path.join(base_dir_model, check_point, f"seed{seed}_fold{fold+1}_stage1")
        if not os.path.exists(stage1_dir):
            print(f" Stage-1 directory not found: {stage1_dir}, skipping...")
            continue

        print(f" Loading Stage-1 model from: {stage1_dir}")
        peft_model_stage1 = PeftModel.from_pretrained(base_model, stage1_dir)

        X_train = df.iloc[train_idx].copy()
        X_eval = df.iloc[val_idx].copy()
        X_test = pd.DataFrame(X_eval.apply(generate_test_prompt, axis=1), columns=["text"])
        X_train["text"] = X_train.apply(generate_prompt, axis=1)
        X_eval["text"] = X_eval.apply(generate_prompt, axis=1)
        y_true = X_eval["y"].tolist()

        y_pred, attention_outputs = predict_with_attention_rollout(
            X_test, peft_model_stage1, tokenizer, device, visualize=True
        )

        cm_stage1 = confusion_matrix(y_true, y_pred, labels=["yes","no"])
        f1_stage1 = f1_score(y_true, y_pred, labels=["yes","no"], average=None, zero_division=0)[0]
        print(f" Stage-1 F1 (yes): {f1_stage1:.4f}")

        stage2_dir = os.path.join(
            base_dir_model,
            check_point,
            f"seed{seed}_fold{fold+1}_stage1",
            "refined_ft",
            f"seed{seed}_fold{fold+1}_stage2_added_guideline"
        )
        if not os.path.exists(stage2_dir):
            print(f" Stage-2 directory not found: {stage2_dir}, skipping Stage-2...")
            continue

        print(f" Loading Stage-2 model from: {stage2_dir}")
        peft_model_stage2 = PeftModel.from_pretrained(base_model, stage2_dir)

        stage2_df = pd.DataFrame({
            "summary": X_eval["summary_en"].tolist(),
            "pred1": y_pred,
            "true": y_true
        })
        stage2_df["text"] = stage2_df.apply(
            lambda r: build_reconsider_prompt(
                summary=r["summary"],
                pred=r["pred1"],
                true=r["true"],
                reveal_true=False
            ),
            axis=1
        )

        y_pred_stage2, attention_outputs_stage2 = predict_with_attention_rollout(
            stage2_df[["text"]], peft_model_stage2, tokenizer, device, visualize=True
        )

        cm_stage2 = confusion_matrix(y_true, y_pred_stage2, labels=["yes","no"])
        f1_stage2 = f1_score(y_true, y_pred_stage2, labels=["yes","no"], average=None, zero_division=0)[0]
        print(f" Stage-2 F1 (yes): {f1_stage2:.4f}")

        output_data_stage1 = {
            "attention_outputs": attention_outputs,
            "summaries": X_eval["summary_en"].tolist(),
            "texts": X_eval["text"].tolist(),
            "y_true": y_true,
            "y_pred": y_pred,
        }
        save_dir_stage1 = "<STAGE1_OUTPUT_DIR>"
            base_dir_results, "<CROSS_VALIDATION_DIR>",
            "<EXPERIMENT_DIR>",
            "attention_rollouts", "stage_1"
        )
        os.makedirs(save_dir_stage1, exist_ok=True)
        save_path_stage1 = os.path.join(save_dir_stage1, f"seed{seed}_fold{fold+1}_rollout.pkl")
        with open(save_path_stage1, "wb") as f:
            pickle.dump(output_data_stage1, f)
        print(f" Saved stage-1 attention outputs to {save_path_stage1}")

        output_data_stage2 = {
            "attention_outputs": attention_outputs_stage2,
            "summaries": stage2_df["summary"].tolist(),
            "texts": stage2_df["text"].tolist(),
            "y_true": y_true,
            "y_pred_stage1": y_pred,
            "y_pred_stage2": y_pred_stage2,
            "reveal_true_in_prompt": reveal_true_in_prompt,
        }
        save_dir_stage2 = "<STAGE2_OUTPUT_DIR>"
            base_dir_results, "<CROSS_VALIDATION_DIR>",
            "<EXPERIMENT_DIR>",
            "attention_rollouts", "stage_2_crot"
        )
        os.makedirs(save_dir_stage2, exist_ok=True)
        save_path_stage2 = os.path.join(save_dir_stage2, f"seed{seed}_fold{fold+1}_rollout.pkl")
        with open(save_path_stage2, "wb") as f:
            pickle.dump(output_data_stage2, f)
        print(f" Saved stage-2 attention outputs to {save_path_stage2}")

end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\n Total Computation Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
