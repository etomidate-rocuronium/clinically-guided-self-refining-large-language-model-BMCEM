import pandas as pd

file_path = "<DATASET_PATH>"
df = pd.read_pickle(file_path)
df


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time                          
import os
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

                                
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

                                                             
def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)                                   

                                                                               
class TranslationDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def translate_batch(summary_texts):
                                               
    system_prompt = "You are a neurologist who is fluent in both Korean and English. Translate and summarize the code-mixed text into English."
    
                                                   
    prompts = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": "user", "content": summary_text} 
            for summary_text in summary_texts
        ]
    ]
    
                                     
    inputs = tokenizer(
        [prompt['content'] for prompt in prompts],                                                 
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")
    
                                   
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
    
                                
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return translated_texts

def main():
                                                  
    init_distributed()

                                                  
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
                                                     
    model = torch.nn.DataParallel(model)

                                                                                   
    batch_size = 20                                                
    translated_summaries = []

                                                     
    total_start_time = time.time()

                                    
    dataset = TranslationDataset(df['summary_en_preprocessed'].tolist())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                                                              
    for start_idx, batch in enumerate(data_loader):
                                           
        start_time = time.time()

                             
        translated_batch = translate_batch(batch)

                                          
        end_time = time.time()

                                               
        elapsed_time_seconds = end_time - start_time
        hours, remainder = divmod(elapsed_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

                                                                     
        print(f"Batch {start_idx + 1}: Time taken = {int(hours):02}:{int(minutes):02}:{int(seconds):02} (HH:MM:SS)")
        
                                                 
        translated_summaries.extend(translated_batch)

                                          
    total_end_time = time.time()

                                      
    total_elapsed_time_seconds = total_end_time - total_start_time
    total_hours, total_remainder = divmod(total_elapsed_time_seconds, 3600)
    total_minutes, total_seconds = divmod(total_remainder, 60)

                                                 
    print(f"\nTotal Time Taken for All Batches: {int(total_hours):02}:{int(total_minutes):02}:{int(total_seconds):02} (HH:MM:SS)")

                                                           
    df.loc[:len(translated_summaries)-1, 'translated_summary'] = translated_summaries

if __name__ == "__main__":
    main()

        

df

        

                              

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time
import os
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

                                
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)                                   

class TranslationDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def translate_and_summarize(text):
    prompt = f"""
    The following text contains clinical notes in Korean and English. Positive neurological findings are usually followed, not preceded, with (+) or +. Negative neurological findings are usually followed, not preceded, with (-) or -.
    {text}

    I want you to translate and summarize the text to English narrative. Do not make things up.

    """
    return prompt

def translate_batch(summary_texts):
    prompt = [translate_and_summarize(text) for text in summary_texts]

    inputs = tokenizer(
        prompt,                                  
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to("cuda")

                                   
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

                                
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return translated_texts

def main():
                                                          
    init_distributed()

                                          
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",                                                       
    )

                                                          
    model = torch.nn.DataParallel(model)

                                                                                   
    batch_size = 20                                                
    translated_summaries = []

    total_start_time = time.time()                         

                                    
    dataset = TranslationDataset(df['summary_en_preprocessed'].tolist())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

                                                              
    for start_idx, batch in enumerate(data_loader):
                                           
        start_time = time.time()

                             
        translated_batch = translate_batch(batch)

                                          
        end_time = time.time()

                                               
        elapsed_time_seconds = end_time - start_time
        hours, remainder = divmod(elapsed_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

                                                                     
        print(f"Batch {start_idx + 1}: Time taken = {int(hours):02}:{int(minutes):02}:{int(seconds):02} (HH:MM:SS)")

                                                 
        translated_summaries.extend(translated_batch)

                          
    total_end_time = time.time()

                                      
    total_elapsed_time_seconds = total_end_time - total_start_time
    total_hours, total_remainder = divmod(total_elapsed_time_seconds, 3600)
    total_minutes, total_seconds = divmod(total_remainder, 60)

                                                 
    print(f"\nTotal Time Taken: {int(total_hours):02}:{int(total_minutes):02}:{int(total_seconds):02} (HH:MM:SS)")

                                                           
    df.loc[:len(translated_summaries)-1, 'translated_summary_extended_prompt'] = translated_summaries

if __name__ == "__main__":
    main()

df

print("Hooray!")
