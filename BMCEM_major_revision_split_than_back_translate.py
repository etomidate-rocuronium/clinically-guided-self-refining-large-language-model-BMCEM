import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForCausalLM

                                                           
             
                                                           
ALL_SEEDS = [7, 42, 99, 123, 2024]
N_SPLITS = 5
TARGET_PER_CLASS = 1000
BATCH_SIZE = 15

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

                                                           
              
                                                           
df_final = pd.read_pickle("<DATASET_PATH>")
df_final = df_final.reset_index(drop=True)

                                                           
                      
                                                           
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"

print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

max_memory_mapping = {
    0: "11GiB",
    1: "11GiB",
    2: "11GiB",
    3: "11GiB"
}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    max_memory=max_memory_mapping
)

print("Model loaded successfully.")

                                                           
                        
                                                           
def batch_generate(prompts_list, max_new_tokens=1500):
    batch_texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts_list
    ]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_length:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    del inputs, outputs, generated_tokens
    torch.cuda.empty_cache()

    return decoded

                                                           
                     
                                                           
def process_dataframe_in_batches(df, column_name, batch_size=4):
    texts = df[column_name].tolist()
    augmented_texts = []

    sys_prompt_ko = "You translate English medical text into Korean. Return ONLY the translation."
    sys_prompt_en = "You translate Korean medical text into English. Return ONLY the translation."

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Batch (BS={batch_size})"):
        batch_raw = texts[i: i + batch_size]

                 
        prompts_ko = [[
            {"role": "system", "content": sys_prompt_ko},
            {"role": "user", "content": text}
        ] for text in batch_raw]

        try:
            korean_translations = batch_generate(prompts_ko)
        except Exception as e:
            print(f"\nError in batch {i} (EN→KO): {e}")
            augmented_texts.extend(batch_raw)
            torch.cuda.empty_cache()
            continue

                 
        prompts_en = [[
            {"role": "system", "content": sys_prompt_en},
            {"role": "user", "content": k_text}
        ] for k_text in korean_translations]

        try:
            english_back = batch_generate(prompts_en)
            augmented_texts.extend(english_back)
        except Exception as e:
            print(f"\nError in batch {i} (KO→EN): {e}")
            augmented_texts.extend(batch_raw)

        gc.collect()
        torch.cuda.empty_cache()

    return augmented_texts

                                                           
              
                                                           
for seed in ALL_SEEDS:
    print(f"\n\n############################")
    print(f"### SEED {seed}")
    print(f"############################")

    set_seed(seed)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_final, df_final["y"])):
        print(f"\n========== SEED {seed} | FOLD {fold+1} ==========")

                                   
               
                                   
        df_train = df_final.iloc[train_idx].copy().reset_index(drop=True)
        df_val = df_final.iloc[val_idx].copy().reset_index(drop=True)

                                   
                                               
                                   
        df_class0 = df_train[df_train["y"] == 0].copy()
        df_class1 = df_train[df_train["y"] == 1].copy()

        n0, n1 = len(df_class0), len(df_class1)
        print(f"Before → Class 0: {n0}, Class 1: {n1}")

        if n0 > n1:
            majority_df = df_class0
            minority_df = df_class1
            minority_label = 1
        else:
            majority_df = df_class1
            minority_df = df_class0
            minority_label = 0

                                       
        df_majority = majority_df.sample(
            n=TARGET_PER_CLASS,
            replace=False,
            random_state=seed
        ).reset_index(drop=True)

                                  
        if len(minority_df) >= TARGET_PER_CLASS:
            df_minority = minority_df.sample(
                n=TARGET_PER_CLASS,
                replace=False,
                random_state=seed
            ).reset_index(drop=True)
        else:
            needed = TARGET_PER_CLASS - len(minority_df)
            print(f"Minority ({minority_label}) needs {needed} samples")

            df_sampled = minority_df.sample(
                n=needed,
                replace=True,
                random_state=seed
            ).reset_index(drop=True)

            augmented_texts = process_dataframe_in_batches(
                df_sampled,
                "translated_summary_cleaned",
                batch_size=BATCH_SIZE
            )

            df_aug = df_sampled.copy()
            df_aug["translated_summary_cleaned"] = augmented_texts
            df_aug["augmented"] = True

            df_minority_full = pd.concat([minority_df, df_aug], ignore_index=True)

            df_minority = df_minority_full.sample(
                n=TARGET_PER_CLASS,
                random_state=seed
            ).reset_index(drop=True)

        df_majority["augmented"] = False
        df_minority["augmented"] = df_minority.get("augmented", False)

        df_train_final = pd.concat([df_majority, df_minority], ignore_index=True)

                 
        df_train_final = df_train_final.sample(
            frac=1,
            random_state=seed
        ).reset_index(drop=True)

                      
        counts = df_train_final["y"].value_counts()
        print(f"After → {counts.to_dict()}")
        assert counts[0] == 1000 and counts[1] == 1000
        assert len(df_train_final) == 2000

                                   
                                
                                   
        print(f"Validation distribution → {df_val['y'].value_counts().to_dict()}")

                                   
              
                                   
        train_path = f"intermediate/train_seed{seed}_fold{fold}.pkl"
        val_path = f"intermediate/val_seed{seed}_fold{fold}.pkl"

        df_train_final.to_pickle(train_path)
        df_val.to_pickle(val_path)

        print(f"Saved: {train_path}")
        print(f"Saved: {val_path}")

        gc.collect()
        torch.cuda.empty_cache()

print("Hooray!")

df_exp = pd.read_pickle("<DATASET_PATH>")

df_exp

df_exp['y'].value_counts()
