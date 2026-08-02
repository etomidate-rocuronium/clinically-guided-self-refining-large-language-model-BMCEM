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

df_final = pd.read_pickle("<DATASET_PATH>")

df_final

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

df = pd.read_pickle("<DATASET_PATH>")

            
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for _, row in df.iterrows():
    reference = row['summary_en'].split()
    candidate = row['translated_summary_cleaned'].split()
    
          
    bleu = sentence_bleu([reference], candidate, smoothing_function=smooth)
    bleu_scores.append(bleu)
    
           
    scores = rouge.score(row['summary_en'], row['translated_summary_cleaned'])
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

df['BLEU'] = bleu_scores
df['ROUGE-1'] = rouge1_scores
df['ROUGE-2'] = rouge2_scores
df['ROUGE-L'] = rougeL_scores

       
print("Average BLEU:", df['BLEU'].mean())
print("Average ROUGE-1:", df['ROUGE-1'].mean())
print("Average ROUGE-2:", df['ROUGE-2'].mean())
print("Average ROUGE-L:", df['ROUGE-L'].mean())

      
df.to_csv("scored_translations.csv", index=False)

print("Hooray!")
