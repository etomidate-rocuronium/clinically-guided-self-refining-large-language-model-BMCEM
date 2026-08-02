import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from peft import PeftModel
from tqdm import tqdm

                                                              
            
                                                              
import os
import sys
import time
import pickle
import torch
import pandas as pd
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

                           
                                       
                           
try:
    import numpy.core.numeric as _num
    sys.modules['numpy._core'] = sys.modules['numpy.core']
    sys.modules['numpy._core.numeric'] = _num
except ImportError:
    if hasattr(np, '_core'):
        sys.modules['numpy.core'] = np._core
        sys.modules['numpy.core.numeric'] = np._core.numeric

                                 
local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank >= 0:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
else:
    world_size = 1
    rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if rank == 0: print(f" Initializing Causal LOO on {world_size} GPU(s)...")

                           
                
                           
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

YES_ID = tokenizer.convert_tokens_to_ids("yes")
NO_ID = tokenizer.convert_tokens_to_ids("no")

def predict_with_loo_causal_logprob(test_df, model, tokenizer, device, eps=1e-12):

    y_pred = []
    importance_outputs = []
    model.eval()

    for i in tqdm(range(len(test_df)), desc=f"Rank {rank} LOO (logP)"):

        raw_text = test_df.iloc[i]["text"]

                                   
                                       
                                   
        start_idx = raw_text.lower().find("patient summary:")
        end_idx = raw_text.lower().find("true label:")

        if start_idx == -1:
            start_idx = 0
        else:
            start_idx += len("patient summary:")

        if end_idx == -1:
            end_idx = len(raw_text)

        clinical_summary = raw_text[start_idx:end_idx].strip()

                                   
        for md in ["```", "#", "markdown"]:
            clinical_summary = clinical_summary.replace(md, "")
        clinical_summary = clinical_summary.strip()

                                   
                  
                                   
        inputs = tokenizer(clinical_summary, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

                                   
                                                          
                                   
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))                       
            base_logits = outputs.logits

                                            
            pred_id = torch.argmax(base_logits[0, -1, :]).item()
            pred_text = tokenizer.decode([pred_id]).strip().lower()
            final_pred = "yes" if "yes" in pred_text else "no"
            y_pred.append(final_pred)

            target_id = YES_ID if final_pred == "yes" else NO_ID

                                                      
            base_log_probs = F.log_softmax(base_logits, dim=-1)                   
            base_mean_logp = base_log_probs[:, :, target_id].mean().item()

                                   
                         
                                   
        token_importances = np.zeros(len(tokens), dtype=np.float32)

        for idx in range(len(tokens)):
            if not tokens[idx].strip():
                continue

            shorn_input_ids = torch.cat([input_ids[:idx], input_ids[idx+1:]]).unsqueeze(0)

            with torch.no_grad():
                out_shorn = model(shorn_input_ids)
                shorn_log_probs = F.log_softmax(out_shorn.logits, dim=-1)
                shorn_mean_logp = shorn_log_probs[:, :, target_id].mean().item()

                                                                                  
            token_importances[idx] = base_mean_logp - shorn_mean_logp

                                   
                      
                                   
        top_indices = np.argsort(token_importances)[-30:][::-1]

        print(f"\n[Rank {rank}] Sample {i} | Pred: {final_pred}")
        print(f"[Rank {rank}] base_mean_logp(target): {base_mean_logp:.6e}")
        print(f"[Rank {rank}] Top 30 causal tokens (Δ logP):")

        for j in top_indices:
            tok = tokens[j].replace("Ġ", " ").replace("Ċ", "\\n").strip()
            if not tok:
                tok = tokens[j]
            print(f"  - '{tok}': {token_importances[j]:.6e}")

               
        cleaned = [
            (
                tokens[k].replace("Ġ", " ").replace("Ċ", "\n").strip() or tokens[k],
                float(token_importances[k]),
            )
            for k in range(len(tokens))
        ]
        importance_outputs.append(cleaned)

    return y_pred, importance_outputs

                           
                    
                           
def generate_test_prompt(data_point):
    return f"""
            You are a neurologist working in the emergency department. Your task is to read the following patient summary and decide if the patient is likely to be diagnosed acute stroke or not. Please respond ONLY with the word 'yes' or 'no'.
            Patient summary: {data_point["summary_en"]}
            True label: """.strip()

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
     
df = pd.read_pickle("<DATASET_PATH>")
df["y"] = df["y"].map({1: "yes", 0: "no"})

def extract_clinical_summary(raw_text: str) -> str:
    start_idx = raw_text.lower().find("patient summary:")
    end_idx = raw_text.lower().find("true label:")

    if start_idx == -1:
        start_idx = 0
    else:
        start_idx += len("patient summary:")

    if end_idx == -1:
        end_idx = len(raw_text)

    summary = raw_text[start_idx:end_idx]
    for md in ["```", "#", "markdown"]:
        summary = summary.replace(md, "")
    return summary.strip()

def get_yes_no_logprobs(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
        logp = F.log_softmax(logits, dim=-1)

    logp_yes = logp[YES_ID].item()
    logp_no = logp[NO_ID].item()
    pred = "yes" if logp_yes > logp_no else "no"
    return pred, logp_yes, logp_no

                                                              
                                   
                                                              

def token_level_loo(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0)).logits
        base_logp = F.log_softmax(logits, dim=-1)
        pred_id = torch.argmax(logits[0, -1]).item()
        pred = "yes" if "yes" in tokenizer.decode([pred_id]).lower() else "no"
        target_id = YES_ID if pred == "yes" else NO_ID
        base_score = base_logp[:, :, target_id].mean().item()

    deltas = np.zeros(len(tokens), dtype=np.float32)

    for i in range(len(tokens)):
        shorn = torch.cat([input_ids[:i], input_ids[i+1:]]).unsqueeze(0)
        with torch.no_grad():
            out = model(shorn).logits
            lp = F.log_softmax(out, dim=-1)
            score = lp[:, :, target_id].mean().item()
        deltas[i] = base_score - score

    cleaned = [
        (tokens[i].replace("Ġ", " ").strip(), float(deltas[i]))
        for i in range(len(tokens))
        if tokens[i].replace("Ġ", "").strip()
    ]

    return pred, cleaned

def plot_token_importance(token_attr, top_k=25):
    tokens, vals = zip(*token_attr)
    vals = np.array(vals)

    idx = np.argsort(np.abs(vals))[-top_k:][::-1]
    tokens = [tokens[i] for i in idx]
    vals = vals[idx]

    colors = ["red" if v > 0 else "blue" for v in vals]

    plt.figure(figsize=(5, 5))
    plt.barh(range(len(tokens)), vals, color=colors)
    plt.gca().invert_yaxis()
    plt.axvline(0)
    plt.xlabel("ΔlogP(target)")
    plt.title("Token-level causal importance")
    plt.yticks(range(len(tokens)), tokens)
    plt.tight_layout()
    plt.show()

                                                              
                           
                                                              

SECTION_RULES = {

                               
                           
                               
    "laterality / focality": [
        r"\bright\b", r"\bleft\b",
        r"\bunilateral\b", r"\bbilateral\b",
        r"\basymmetri\w*\b", r"\bsymmetric\b",
        r"\bhemipare\w*\b", r"\bhemipleg\w*\b",
        r"\bmonopare\w*\b", r"\bmonopleg\w*\b",
        r"\bfocal\b", r"\bfocally\b"
    ],

                               
                           
                               
    "neurological deficits": [
        r"\bweakness\b", r"\bparesis\b", r"\bparalysis\b",
        r"\bnumb\w*\b", r"\btingl\w*\b", r"\bparesthes\w*\b",
        r"\bsensation\b", r"\bsensory\b",
        r"\baphasia\b", r"\bdysarthria\b", r"\bdysphasia\b",
        r"\bfacial\b", r"\bfacial\s+droop\b",
        r"\bataxia\b", r"\bdysmetria\b",
        r"\bgait\b", r"\bimbalance\b",
        r"\bvisual\b", r"\bvision\b", r"\bhemianopia\b",
        r"\bNIHSS\b"
    ],

                               
                           
                               
    "vascular risk factors": [
        r"\bhypertension\b", r"\bHTN\b",
        r"\bdiabetes\b", r"\bDM\b",
        r"\bdyslipidemia\b", r"\bHLD\b",
        r"\bhyperlipidemia\b",
        r"\bsmok\w*\b", r"\btobacco\b",
        r"\bobesity\b",
        r"\batrial\s*fibrillation\b", r"\bAF\b",
        r"\bcoronary\b", r"\bCAD\b",
        r"\bvascular\b", r"\batherosclero\w*\b"
    ],

                               
                                        
                               
    "seizure / epilepsy": [
        r"\bseizure\b", r"\bseizures\b",
        r"\bepilep\w*\b",
        r"\bpost[-\s]?ictal\b",
        r"\btodd\w*\s+paralysis\b",
        r"\bconvulsion\b"
    ],

                               
                                              
                               
    "cancer": [
        r"\bcancer\b", r"\bmalignan\w*\b",
        r"\bmetasta\w*\b",
        r"\bneoplasm\b",
        r"\btumou?r\b",
        r"\boncology\b", r"\boncologic\w*\b",
        r"\bchemotherap\w*\b", r"\bchemo\b",
        r"\bradiation\b", r"\bradiotherap\w*\b",
        r"\bhypercoagul\w*\b",
        r"\bparaneoplastic\b"
    ],

                               
                                      
                               
    "anti-platelet therapy": [
        r"\baspirin\b", r"\bASA\b",
        r"\bclopidogrel\b", r"\bplavix\b",
        r"\bwarfarin\b", r"\bcoumadin\b",
        r"\bapixaban\b", r"\beliquis\b",
        r"\brivaroxaban\b", r"\bxarelto\b",
        r"\bdabigatran\b", r"\bpradaxa\b",
        r"\bheparin\b",
        r"\banticoag\w*\b", r"\bantiplatelet\b"
    ],

                               
                                            
                               
    "temporal course": [
        r"\bacute\b", r"\bacutely\b",
        r"\bsudden\b", r"\babrupt\b",
        r"\bonset\b",
        r"\bprogressive\b", r"\bgradual\b",
        r"\bchronic\b",
        r"\bworsen\w*\b", r"\bimprov\w*\b",
        r"\bhistory\s+of\b"
    ],
}

def ablate_text(text, patterns):
    out = text
    for p in patterns:
        out = re.sub(p, "[REMOVED]", out, flags=re.IGNORECASE)
    return out

def section_level_ablation(prompt, model, tokenizer, device):
    base_pred, logp_yes, logp_no = get_yes_no_logprobs(prompt, model, tokenizer, device)
    base_score = logp_yes if base_pred == "yes" else logp_no

    deltas = {}
    for name, pats in SECTION_RULES.items():
        ablated = ablate_text(prompt, pats)
        _, y2, n2 = get_yes_no_logprobs(ablated, model, tokenizer, device)
        score2 = y2 if base_pred == "yes" else n2
        deltas[name] = base_score - score2

    return base_pred, deltas

def plot_section_importance(deltas, base_pred):
    names = list(deltas.keys())
    vals = np.array(list(deltas.values()))

    order = np.argsort(np.abs(vals))[::-1]
    names = [names[i] for i in order]
    vals = vals[order]

    colors = ["red" if v > 0 else "blue" for v in vals]

    plt.figure(figsize=(5, 5))
    plt.barh(range(len(names)), vals, color=colors)
    plt.gca().invert_yaxis()
    plt.axvline(0)
    plt.xlabel("ΔlogP(target)")
    plt.title(f"Section-level causal importance (baseline={base_pred})")
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    plt.show()

def visualize_causal_heatmap_token_level(token_importances, figsize=(5, 5)):

    tokens, delta_logp = zip(*token_importances)

    plt.figure(figsize=figsize)
    plt.bar(range(len(tokens)), delta_logp, color="salmon")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.ylabel("Δ log P(target)")
    plt.title("Token-level Causal Importance (ΔlogP)")
    plt.tight_layout()
    plt.show()

                                                              
                     
                                                              

def run_full_interpretability(
    df,
    seed,
    fold,
    patient_idx,
    stage,
    base_model,
    tokenizer,
    device,
    base_dir_model,
    check_point,
):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    _, val_idx = list(skf.split(df, df["y"]))[fold - 1]
    X_val = df.iloc[val_idx].reset_index(drop=True)

    patient = X_val.iloc[patient_idx]

    stage1_path = os.path.join(
        base_dir_model, check_point, f"seed{seed}_fold{fold}_stage1"
    )
    peft_s1 = PeftModel.from_pretrained(base_model, stage1_path)

    prompt_s1 = generate_test_prompt(patient)
    pred1, _, _ = get_yes_no_logprobs(
        extract_clinical_summary(prompt_s1), peft_s1, tokenizer, device
    )

    if stage == "stage2":
        stage2_path = os.path.join(
            stage1_path, "refined_ft",
            f"seed{seed}_fold{fold}_stage2_added_guideline"
        )
        peft_model = PeftModel.from_pretrained(base_model, stage2_path)
        prompt = build_reconsider_prompt(patient["summary_en"], pred1)
    else:
        peft_model = peft_s1
        prompt = prompt_s1

    clinical = extract_clinical_summary(prompt)

    print("\n=== TOKEN-LEVEL LOO ===")
    pred, token_attr = token_level_loo(clinical, peft_model, tokenizer, device)
    visualize_causal_heatmap_token_level(token_attr)
    plot_token_importance(token_attr)

    print("\n=== SECTION-LEVEL ABLATION ===")
    base_pred, deltas = section_level_ablation(clinical, peft_model, tokenizer, device)
    plot_section_importance(deltas, base_pred)

    return {
        "seed": seed,
        "fold": fold,
        "patient_idx": patient_idx,
        "prediction": base_pred,
        "token_attr": token_attr,
        "section_attr": deltas,
        "summary": clinical,
    }

result = run_full_interpretability(
    df=df,
    seed=123,
    fold=1,
    patient_idx=0,
    stage="stage2",
    base_model=base_model,
    tokenizer=tokenizer,
    device=device,
    base_dir_model="<MODEL_DIR>"
    check_point="<CHECKPOINT_DIR>"
)

token_attr = result["token_attr"]
section_attr = result["section_attr"]

from IPython.display import display, HTML
import pickle
import numpy as np
import ipywidgets as widgets
import glob
import os

                                
save_dir = "./results/llama3_8b/attention_rollouts/stage_2_crot"

                          
pkl_files = sorted(glob.glob(os.path.join(save_dir, "seed*_fold*.pkl")))

if not pkl_files:
    raise FileNotFoundError("No seed/fold files found in the directory.")

                                           
file_options = []
for path in pkl_files:
    fname = os.path.basename(path)
    parts = fname.replace(".pkl", "").split("_")
    seed = int(parts[0].replace("seed",""))
    fold = int(parts[1].replace("fold",""))
    file_options.append((f"Seed {seed} | Fold {fold}", path))

                        
file_selector = widgets.Dropdown(
    options=file_options,
    description="Seed/Fold:",
    layout=widgets.Layout(width="350px"),
    style={'description_width':'initial'}
)

                                
index_selector = widgets.Dropdown(
    options=[],
    description="Patient index:",
    layout=widgets.Layout(width="300px"),
    style={'description_width':'initial'}
)

output_area = widgets.Output()

def debug_trailing_tokens(tokens_and_scores, n=20):
    print(f"Last {n} tokens for inspection:")
    for i, (tok, _) in enumerate(tokens_and_scores[-n:], start=len(tokens_and_scores)-n):
        print(f"{i}: {repr(tok)}")

def trim_tokens_up_to_period(tokens_and_scores, n_start=50):
    trimmed = tokens_and_scores[n_start:]
    period_tokens = {'.', '.Ċ'}
    last_period_idx = None
    for i in reversed(range(len(trimmed))):
        tok = trimmed[i][0]
        if tok in period_tokens:
            last_period_idx = i
            break
    if last_period_idx is not None:
        trimmed = trimmed[:last_period_idx+1]
    if len(trimmed)>0:
        trimmed = trimmed[:-1]
    return trimmed

def color_token_html(token, normalized_score):
    clean_token = token.replace("▁"," ").replace("Ġ"," ")
    r = int(50 + (255 - 5)*normalized_score)
    g = int(50*(1 - normalized_score))
    b = int(50*(1 - normalized_score))
    return f'<span style="background-color: rgb({r},{g},{b}); padding:1px; margin:1px;">{clean_token}</span>'

def load_and_prepare(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

                                   
loaded_data = {}

def on_file_change(change):
    path = change['new']
    if path not in loaded_data:
        data = load_and_prepare(path)
        loaded_data[path] = data
    else:
        data = loaded_data[path]
    
    n = len(data["attention_outputs"])
    index_selector.unobserve(on_index_change, names="value")
    index_selector.options = list(range(n))
    index_selector.value = 0
    index_selector.observe(on_index_change, names="value")
    
                                         
    show_output(path, 0)

def on_index_change(change):
    idx = change['new']
    path = file_selector.value
    show_output(path, idx)

def show_output(path, idx):
    data = loaded_data[path]
    attn = data["attention_outputs"]
    y_true = data["y_true"]
    y_pred = data["y_pred_stage2"]
    
    tokens_and_scores = attn[idx]
    filtered = trim_tokens_up_to_period(tokens_and_scores)
    
    scores = [s if not isinstance(s,list) else sum(s)/len(s) for _,s in filtered]
    min_s = min(scores)
    max_s = max(scores)
    if abs(max_s - min_s)<1e-8:
        max_s += 1e-8
    normalized = [(s - min_s)/(max_s - min_s) for s in scores]
    
    colored_tokens = [
        color_token_html(tok, n)
        for (tok,_),n in zip(filtered, normalized)
    ]
    
    with output_area:
        output_area.clear_output()
        print(f" Patient {idx}")
        print(f"True label : {y_true[idx]}")
        print(f"Predicted label : {y_pred[idx]}\n")
        display(HTML(f"<div style='line-height:1.6;'>{''.join(colored_tokens)}</div>"))

                 
file_selector.observe(on_file_change, names="value")
index_selector.observe(on_index_change, names="value")

                 
display(file_selector)
display(index_selector)
display(output_area)

                       
on_file_change({'new': file_selector.value})

from IPython.display import display, HTML
import numpy as np
import matplotlib.pyplot as plt
import os

                           
                           
                           
PATH = file_selector.value
IDX  = index_selector.value

TITLE = "Stage-2 c-RoT Attention Rollout (Clinical Summary Only)"
FONT_FAMILY = "Times New Roman, serif"
MAX_WIDTH_PX = 1100

SECTION_MARKERS = [
    "Clinical Summary",
    "Clinical summary",
    "Summary of present illness",
]

EPS = 1e-12
ABS_SUPPRESS = 1e-10

                           
                                      
                           
def trim_before_marker(tokens_and_scores, markers):
    tokens = [t for t, _ in tokens_and_scores]
    clean = [t.replace("▁"," ").replace("Ġ"," ") for t in tokens]
    full_text = "".join(clean).lower()

    marker_pos = None
    for m in markers:
        pos = full_text.find(m.lower())
        if pos != -1:
            marker_pos = pos
            break

    if marker_pos is None:
        return tokens_and_scores, False

    running_len = 0
    start_idx = 0
    for i, tok in enumerate(clean):
        running_len += len(tok)
        if running_len >= marker_pos:
            start_idx = i
            break

    return tokens_and_scores[start_idx:], True

def quantile_rescale(scores, q_low=0.05, q_high=0.95, eps=1e-12):
    s = np.array(scores, dtype=float)
    if np.any(s > 0):
        lo = np.quantile(s[s > 0], q_low)
    else:
        lo = eps
    hi = np.quantile(s, q_high)
    hi = max(hi, lo + eps)
    s_clipped = np.clip(s, lo, hi)
    return (s_clipped - lo) / (hi - lo)

def sum_normalize(scores):
    s = np.array(scores, dtype=float)
    total = s.sum()
    if total < EPS:
        return np.zeros_like(s)
    return s / total

def log_rescale(scores):
    s = np.maximum(scores, EPS)
    log_s = np.log10(s)
    denom = (log_s.max() - log_s.min())
    if denom < EPS:
        return np.zeros_like(log_s)
    return (log_s - log_s.min()) / denom

def token_html(token, vis_score):
    clean = token.replace("▁"," ").replace("Ġ"," ")
    r = 255
    g = int(255 * (1 - vis_score))
    b = int(255 * (1 - vis_score))
    return f"""
    <span style="
        background-color: rgb({r},{g},{b});
        padding:2px 3px;
        margin:1px;
        border-radius:3px;
        display:inline-block;
    ">{clean}</span>
    """

                           
                          
                           
def plot_token_importance_ax(ax, token_attr, top_k=30):

    tokens = [t for t, _ in token_attr]
    vals = np.array([v for _, v in token_attr], dtype=float)

                             
    k = min(top_k, len(vals))
    idx = np.argsort(np.abs(vals))[-k:][::-1]

    tokens_k = [tokens[i] for i in idx]
    vals_k = vals[idx]

    colors = ["red" if v > 0 else "blue" for v in vals_k]

    ax.barh(range(k), vals_k, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(k))
    ax.set_yticklabels(tokens_k, fontsize=8)
    ax.set_xlabel("ΔlogP(target)")
    ax.set_title(f"Token level (Top {k})")

def plot_section_importance_ax(ax, section_attr, baseline_pred="yes"):

    names = list(section_attr.keys())
    vals = np.array([section_attr[n] for n in names], dtype=float)

    order = np.argsort(np.abs(vals))[::-1]
    names = [names[i] for i in order]
    vals = vals[order]

    colors = ["red" if v > 0 else "blue" for v in vals]

    ax.barh(range(len(names)), vals, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ΔlogP(target)")
    ax.set_title(f"Section level (baseline prediction = {baseline_pred})")

                                                              
                                            
                                                              
data = loaded_data[PATH]
tokens_and_scores = data["attention_outputs"][IDX]
y_true = data["y_true"][IDX]
y_pred = data["y_pred_stage2"][IDX]
correct = "✓ Correct" if y_true == y_pred else "✗ Incorrect"

trimmed_tokens, found_marker = trim_before_marker(tokens_and_scores, SECTION_MARKERS)

                            
raw_scores = []
for _, s in trimmed_tokens:
    if isinstance(s, list):
        raw_scores.append(float(np.mean(s)))
    else:
        raw_scores.append(float(s))

raw_scores = np.array(raw_scores)

                          
attn_mass = sum_normalize(raw_scores)
attn_mass[attn_mass < ABS_SUPPRESS] = 0.0

                                                
vis_scores = quantile_rescale(log_rescale(attn_mass))

colored_tokens = [
    token_html(tok, s)
    for (tok, _), s in zip(trimmed_tokens, vis_scores)
]

marker_note = "" if found_marker else "<i>(Clinical Summary marker not found; full text shown)</i>"

html = f"""
<div style="font-family:{FONT_FAMILY}; max-width:{MAX_WIDTH_PX}px;">

<h3 style="margin-bottom:6px;">{TITLE}</h3>

<div style="font-size:14px; margin-bottom:10px;">
<b>Seed/Fold:</b> {os.path.basename(PATH).replace('.pkl','')} &nbsp; | &nbsp;
<b>Patient index:</b> {IDX} <br>
<b>True label:</b> {y_true} &nbsp; | &nbsp;
<b>Prediction:</b> {y_pred} &nbsp; | &nbsp;
<b>Status:</b> {correct} <br>
{marker_note}
</div>

<div style="
    border:1px solid #ccc;
    padding:12px;
    line-height:1.8;
    font-size:16px;
    background:#ffffff;
">
{''.join(colored_tokens)}
</div>

<div style="margin-top:12px; font-size:13px;">
<b>Color encoding:</b> log-scaled attention mass (sum-normalized) <br>
<span style="background:rgb(255,255,255); padding:2px 6px; border:1px solid #ccc;">Low</span>
→
<span style="background:rgb(255,180,180); padding:2px 6px;">Medium</span>
→
<span style="background:rgb(255,0,0); padding:2px 6px; color:white;">High</span>
</div>

</div>
"""
display(HTML(html))

                                                              
                                                    
                                                              
                                                 
                                            
                                                   
 
                                                             
          
                                       
                                         

assert "token_attr" in globals(), " token_attr is not defined. Compute/load it first."
assert "section_attr" in globals(), " section_attr is not defined. Compute/load it first."

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

plot_token_importance_ax(axes[0], token_attr, top_k=30)
plot_section_importance_ax(axes[1], section_attr, baseline_pred=y_pred)

plt.tight_layout()
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

                                                              
        
                                                              
SAVE_DIR = "./results/llama3_8b/dashboard_exports"
os.makedirs(SAVE_DIR, exist_ok=True)

PATH = file_selector.value
IDX  = index_selector.value

data = loaded_data[PATH]
tokens_and_scores = data["attention_outputs"][IDX]
y_true = data["y_true"][IDX]
y_pred = data["y_pred_stage2"][IDX]
correct = (y_true == y_pred)

seedfold_name = os.path.basename(PATH).replace(".pkl", "")

                                                          
                                       
                    
assert "token_attr" in globals(), " token_attr not found. Compute/load token_attr first."
assert "section_attr" in globals(), " section_attr not found. Compute/load section_attr first."

                                                              
                                                     
                                                              
SECTION_MARKERS = ["Clinical Summary", "Clinical summary", "Summary of present illness"]
EPS = 1e-12
ABS_SUPPRESS = 1e-10

def trim_before_marker(tokens_and_scores, markers):
    tokens = [t for t, _ in tokens_and_scores]
    clean = [t.replace("▁", " ").replace("Ġ", " ") for t in tokens]
    full_text = "".join(clean).lower()

    marker_pos = None
    for m in markers:
        pos = full_text.find(m.lower())
        if pos != -1:
            marker_pos = pos
            break

    if marker_pos is None:
        return tokens_and_scores

    running_len = 0
    start_idx = 0
    for i, tok in enumerate(clean):
        running_len += len(tok)
        if running_len >= marker_pos:
            start_idx = i
            break

    return tokens_and_scores[start_idx:]

def sum_normalize(scores):
    s = np.array(scores, dtype=float)
    total = s.sum()
    if total < EPS:
        return np.zeros_like(s)
    return s / total

def log_rescale(scores):
    s = np.maximum(scores, EPS)
    log_s = np.log10(s)
    denom = (log_s.max() - log_s.min())
    if denom < EPS:
        return np.zeros_like(log_s)
    return (log_s - log_s.min()) / denom

def quantile_rescale(scores, q_low=0.05, q_high=0.95, eps=1e-12):
    s = np.array(scores, dtype=float)
    if np.any(s > 0):
        lo = np.quantile(s[s > 0], q_low)
    else:
        lo = eps
    hi = np.quantile(s, q_high)
    hi = max(hi, lo + eps)
    s_clipped = np.clip(s, lo, hi)
    return (s_clipped - lo) / (hi - lo)

                                                              
                                            
                                                              
def plot_attention_overlay_matplotlib(ax, tokens_and_scores, max_tokens=350):

    trimmed = trim_before_marker(tokens_and_scores, SECTION_MARKERS)

                         
    raw_scores = []
    toks = []
    for tok, s in trimmed[:max_tokens]:
        toks.append(tok.replace("▁", " ").replace("Ġ", " "))
        if isinstance(s, list):
            raw_scores.append(float(np.mean(s)))
        else:
            raw_scores.append(float(s))

    raw_scores = np.array(raw_scores, dtype=float)

    attn_mass = sum_normalize(raw_scores)
    attn_mass[attn_mass < ABS_SUPPRESS] = 0.0

    vis_scores = quantile_rescale(log_rescale(attn_mass))

                                      
    ax.axis("off")

    x, y = 0.01, 0.95
    line_height = 0.13
    space = 0.004

    for tok, s in zip(toks, vis_scores):
        if tok.strip() == "":
            continue

                                      
        rgba = (1.0, 1.0 - s, 1.0 - s, 1.0)

                                                   
        token_width = 0.008 * max(len(tok), 1)

        if x + token_width > 0.99:
            x = 0.01
            y -= line_height
            if y < 0.05:
                break

        ax.text(
            x, y, tok,
            fontsize=10,
            fontfamily="serif",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=rgba, edgecolor="none")
        )
        x += token_width + space

                                                              
                           
                                                              
def plot_token_importance_ax(ax, token_attr, top_k=30):
    tokens = [t for t, _ in token_attr]
    vals = np.array([v for _, v in token_attr], dtype=float)

    k = min(top_k, len(vals))
    idx = np.argsort(np.abs(vals))[-k:][::-1]
    tokens_k = [tokens[i] for i in idx]
    vals_k = vals[idx]

    colors = ["red" if v > 0 else "blue" for v in vals_k]

    ax.barh(range(k), vals_k, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(k))
    ax.set_yticklabels(tokens_k, fontsize=8)
    ax.set_xlabel("ΔlogP(target)")
    ax.set_title(f"Token-level (Top {k})")

                                                              
                      
                                                              
def plot_section_importance_ax(ax, section_attr, baseline_pred="yes"):
    names = list(section_attr.keys())
    vals = np.array([section_attr[n] for n in names], dtype=float)

    order = np.argsort(np.abs(vals))[::-1]
    names = [names[i] for i in order]
    vals = vals[order]

    colors = ["red" if v > 0 else "blue" for v in vals]

    ax.barh(range(len(names)), vals, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ΔlogP(target)")
    ax.set_title(f"Section-level (baseline={baseline_pred})")

                                                              
                                
                                                              
fig = plt.figure(figsize=(16, 15))

                                                       
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])

                          
ax0 = fig.add_subplot(gs[0, :])
ax0.set_title(
    f"Stage-2 Attention Rollout (Matplotlib) | {seedfold_name} | Patient={IDX}\n"
    f"True={y_true} | Pred={y_pred} | {'✓ Correct' if correct else '✗ Incorrect'}",
    fontsize=13,
    fontweight="bold"
)
plot_attention_overlay_matplotlib(ax0, tokens_and_scores, max_tokens=800)

                        
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])

plot_token_importance_ax(ax1, token_attr, top_k=30)
plot_section_importance_ax(ax2, section_attr, baseline_pred=y_pred)

plt.tight_layout()

save_path = os.path.join(SAVE_DIR, f"{seedfold_name}_patient{IDX}_dashboard.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f" Saved dashboard: {save_path}")

import os
import numpy as np
import matplotlib.pyplot as plt

                                                              
        
                                                              
SAVE_DIR = "./results/llama3_8b/dashboard_exports"
os.makedirs(SAVE_DIR, exist_ok=True)

PATH = file_selector.value
IDX  = index_selector.value

data = loaded_data[PATH]
tokens_and_scores = data["attention_outputs"][IDX]
y_true = data["y_true"][IDX]
y_pred = data["y_pred_stage2"][IDX]
correct = (y_true == y_pred)

seedfold_name = os.path.basename(PATH).replace(".pkl", "")

                                                         
assert "token_attr" in globals(), "token_attr not found. Compute/load token_attr first."
assert "section_attr" in globals(), " section_attr not found. Compute/load section_attr first."

                                                              
                                      
                                                              
SECTION_MARKERS = ["Clinical Summary", "Clinical summary", "Summary of present illness"]
EPS = 1e-12
ABS_SUPPRESS = 1e-10

def trim_before_marker(tokens_and_scores, markers):
    tokens = [t for t, _ in tokens_and_scores]
    clean = [t.replace("▁"," ").replace("Ġ"," ") for t in tokens]
    full_text = "".join(clean).lower()

    marker_pos = None
    for m in markers:
        pos = full_text.find(m.lower())
        if pos != -1:
            marker_pos = pos
            break

    if marker_pos is None:
        return tokens_and_scores

    running_len = 0
    start_idx = 0
    for i, tok in enumerate(clean):
        running_len += len(tok)
        if running_len >= marker_pos:
            start_idx = i
            break

    return tokens_and_scores[start_idx:]

def sum_normalize(scores):
    s = np.array(scores, dtype=float)
    total = s.sum()
    if total < EPS:
        return np.zeros_like(s)
    return s / total

def log_rescale(scores):
    s = np.maximum(scores, EPS)
    log_s = np.log10(s)
    denom = (log_s.max() - log_s.min())
    if denom < EPS:
        return np.zeros_like(log_s)
    return (log_s - log_s.min()) / denom

def quantile_rescale(scores, q_low=0.05, q_high=0.95, eps=1e-12):
    s = np.array(scores, dtype=float)
    if np.any(s > 0):
        lo = np.quantile(s[s > 0], q_low)
    else:
        lo = eps
    hi = np.quantile(s, q_high)
    hi = max(hi, lo + eps)
    s_clipped = np.clip(s, lo, hi)
    return (s_clipped - lo) / (hi - lo)

                                                              
                                                      
                                                              
def plot_attention_overlay_chunk(ax, tokens, vis_scores, x_max=0.99, title=None):

    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=2)

    x, y = 0.01, 0.98
    line_height = 0.07
    space = 0.001

    lowest_y = y

    for tok, s in zip(tokens, vis_scores):
        tok = tok.replace("▁", " ").replace("Ġ", " ")
        if tok.strip() == "":
            continue

        rgba = (1.0, 1.0 - s, 1.0 - s, 1.0)
        token_width = 0.006 * max(len(tok), 1)

                                       
        if x + token_width > x_max:
            x = 0.01
            y -= line_height
            lowest_y = min(lowest_y, y)
            if y < 0.02:
                break

        ax.text(
            x, y, tok,
            fontsize=9,
            fontfamily="serif",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=rgba, edgecolor="none")
        )

        x += token_width + space

    ax.set_ylim(lowest_y - 0.05, 1.02)

                                                              
                            
                                                              
def plot_token_importance_ax(ax, token_attr, top_k=30):
    tokens = [t for t, _ in token_attr]
    vals = np.array([v for _, v in token_attr], dtype=float)

    k = min(top_k, len(vals))
    idx = np.argsort(np.abs(vals))[-k:][::-1]
    tokens_k = [tokens[i] for i in idx]
    vals_k = vals[idx]

    colors = ["red" if v > 0 else "blue" for v in vals_k]

    ax.barh(range(k), vals_k, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(k))
    ax.set_yticklabels(tokens_k, fontsize=8)
    ax.set_xlabel("ΔlogP(target)")
    ax.set_title(f"Token-level (Top {k})")

                                                              
                      
                                                              
def plot_section_importance_ax(ax, section_attr, baseline_pred="yes"):
    names = list(section_attr.keys())
    vals = np.array([section_attr[n] for n in names], dtype=float)

    order = np.argsort(np.abs(vals))[::-1]
    names = [names[i] for i in order]
    vals = vals[order]

    colors = ["red" if v > 0 else "blue" for v in vals]

    ax.barh(range(len(names)), vals, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("ΔlogP(target)")
    ax.set_title(f"Section-level (baseline={baseline_pred})")

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

def add_attention_legend_right(fig, x=0.94, y=0.62, w=0.015, h=0.22):

                                   
    ax_leg = fig.add_axes([x, y, w, h])
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.set_frame_on(True)

                                         
    cmap = LinearSegmentedColormap.from_list(
        "attn_red",
        ["#ffffff", "#ffb3b3", "#ff0000"]
    )

    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_leg.imshow(gradient, aspect="auto", cmap=cmap, origin="lower")

                
    fig.text(x + w + 0.01, y, "Low", fontsize=10, va="bottom")
    fig.text(x + w + 0.01, y + h/2, "Med", fontsize=10, va="center")
    fig.text(x + w + 0.01, y + h, "High", fontsize=10, va="top")

    fig.text(x - 0.005, y + h + 0.015, "Attention\n(log-scaled)", fontsize=10, ha="left")

def add_attention_legend_right_edge_aligned(fig, ax_row2_right, y=0.64, h=0.22, w=0.018, pad=0.005):

    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    row2_x1 = ax_row2_right.get_position().x1
    x0 = row2_x1 - w - pad

    ax_leg = fig.add_axes([x0, y, w, h])
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.set_frame_on(True)

    cmap = LinearSegmentedColormap.from_list("attn_red", ["#ffffff", "#ffb3b3", "#ff0000"])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_leg.imshow(grad, aspect="auto", cmap=cmap, origin="lower")

    ax_leg.text(1.25, 1.0, "High", transform=ax_leg.transAxes, va="top", fontsize=10)
    ax_leg.text(1.25, 0.5, "Med",  transform=ax_leg.transAxes, va="center", fontsize=10)
    ax_leg.text(1.25, 0.0, "Low",  transform=ax_leg.transAxes, va="bottom", fontsize=10)

    ax_leg.text(
        -0.7, 1.02, "Attention\n(log-scaled)",
        transform=ax_leg.transAxes,
        ha="left", va="bottom", fontsize=10
    )

    return x0

import matplotlib.pyplot as plt

def plot_attention_overlay_precise(
    fig,
    ax,
    tokens,
    vis_scores,
    x_max=0.99,
    title=None,
    fontsize=9,
    fontfamily="serif",
    pad=0.18,                                    
    x0=0.01,
    y0=0.98,
    line_height=0.08
):

    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=2)

                                           
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    x = x0
    y = y0
    lowest_y = y

    for tok, s in zip(tokens, vis_scores):
        tok = tok.replace("▁", " ").replace("Ġ", " ")
        if tok.strip() == "":
            continue

        rgba = (1.0, 1.0 - s, 1.0 - s, 1.0)

                                                    
        t = ax.text(
            x, y, tok,
            fontsize=fontsize,
            fontfamily=fontfamily,
            va="top",
            ha="left",
            bbox=dict(boxstyle=f"round,pad={pad}", facecolor=rgba, edgecolor="none"),
        )

                                              
        bb = t.get_window_extent(renderer=renderer)

                                               
        ax_bb = ax.get_window_extent(renderer=renderer)
        width_axes = bb.width / ax_bb.width

                                  
        if x + width_axes > x_max:
            t.remove()
            x = x0
            y -= line_height
            lowest_y = min(lowest_y, y)

            if y < 0.02:
                break

                                
            t = ax.text(
                x, y, tok,
                fontsize=fontsize,
                fontfamily=fontfamily,
                va="top",
                ha="left",
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor=rgba, edgecolor="none"),
            )

            fig.canvas.draw()
            bb = t.get_window_extent(renderer=renderer)
            width_axes = bb.width / ax_bb.width

                                                  
        x += width_axes + 0.002            

    ax.set_ylim(lowest_y - 0.05, 1.02)

                                                              
                                      
                                                              
trimmed = trim_before_marker(tokens_and_scores, SECTION_MARKERS)

raw_scores = []
tok_list = []
for tok, s in trimmed:
    tok_list.append(tok)
    if isinstance(s, list):
        raw_scores.append(float(np.mean(s)))
    else:
        raw_scores.append(float(s))

raw_scores = np.array(raw_scores, dtype=float)

attn_mass = sum_normalize(raw_scores)
attn_mass[attn_mass < ABS_SUPPRESS] = 0.0
vis_scores = quantile_rescale(log_rescale(attn_mass))

                     
mid = len(tok_list) // 2
tok_1, tok_2 = tok_list[:mid], tok_list[mid:]
vis_1, vis_2 = vis_scores[:mid], vis_scores[mid:]

                                                              
                                               
                                                              
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[0.38, 0.38, 1.0])

            
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, :])

            
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[2, 1])

                                                          
plot_token_importance_ax(ax2, token_attr, top_k=30)
plot_section_importance_ax(ax3, section_attr, baseline_pred=y_pred)

                                                   
legend_x0 = add_attention_legend_right_edge_aligned(fig, ax3, y=0.64, h=0.22, w=0.018)

                                                  
x_max_overlay = legend_x0                  

                                   
ax0.set_title(
    f"Clinical Summary | {seedfold_name} | Patient Index = {IDX}\n"
    f"Label = {y_true} | Prediction = {y_pred} | {'✓ Correct' if correct else '✗ Incorrect'}",
    fontsize=12,
    fontweight="bold",
    pad=2,
)
plot_attention_overlay_chunk(ax0, tok_1, vis_1, x_max=x_max_overlay)

ax1.set_title("cRoT", fontsize=11, fontweight="bold", pad=2)
plot_attention_overlay_chunk(ax1, tok_2, vis_2, x_max=x_max_overlay)

plt.subplots_adjust(hspace=0.05, top=0.93, bottom=0.06)

save_path = os.path.join(SAVE_DIR, f"{seedfold_name}_patient{IDX}_dashboard_no_overlap.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(" Saved!:", save_path)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

                                                              
                                       
                                                              
TITLE_FONTSIZE = 14
SUBTITLE_FONTSIZE = 12
AXISLABEL_FONTSIZE = 12
TICKLABEL_FONTSIZE = 10
TOKEN_FONTSIZE = 10
LEGEND_FONTSIZE = 11

FONT_FAMILY = "DejaVu Sans"                                                 
FONT_WEIGHT = "normal"

                                                              
        
                                                              
SAVE_DIR = "./results/llama3_8b/dashboard_exports"
os.makedirs(SAVE_DIR, exist_ok=True)

PATH = file_selector.value
IDX  = index_selector.value

data = loaded_data[PATH]
tokens_and_scores = data["attention_outputs"][IDX]
y_true = data["y_true"][IDX]
y_pred = data["y_pred_stage2"][IDX]
correct = (y_true == y_pred)

seedfold_name = os.path.basename(PATH).replace(".pkl", "")

                                                         
assert "token_attr" in globals(), "token_attr not found. Compute/load token_attr first."
assert "section_attr" in globals(), "section_attr not found. Compute/load section_attr first."

                                                              
                                      
                                                              
SECTION_MARKERS = ["Clinical Summary", "Clinical summary", "Summary of present illness"]
EPS = 1e-12
ABS_SUPPRESS = 1e-10

def trim_before_marker(tokens_and_scores, markers):
    tokens = [t for t, _ in tokens_and_scores]
    clean = [t.replace("▁"," ").replace("Ġ"," ") for t in tokens]
    full_text = "".join(clean).lower()

    marker_pos = None
    for m in markers:
        pos = full_text.find(m.lower())
        if pos != -1:
            marker_pos = pos
            break

    if marker_pos is None:
        return tokens_and_scores

    running_len = 0
    start_idx = 0
    for i, tok in enumerate(clean):
        running_len += len(tok)
        if running_len >= marker_pos:
            start_idx = i
            break

    return tokens_and_scores[start_idx:]

def sum_normalize(scores):
    s = np.array(scores, dtype=float)
    total = s.sum()
    if total < EPS:
        return np.zeros_like(s)
    return s / total

def log_rescale(scores):
    s = np.maximum(scores, EPS)
    log_s = np.log10(s)
    denom = (log_s.max() - log_s.min())
    if denom < EPS:
        return np.zeros_like(log_s)
    return (log_s - log_s.min()) / denom

def quantile_rescale(scores, q_low=0.05, q_high=0.95, eps=1e-12):
    s = np.array(scores, dtype=float)
    if np.any(s > 0):
        lo = np.quantile(s[s > 0], q_low)
    else:
        lo = eps
    hi = np.quantile(s, q_high)
    hi = max(hi, lo + eps)
    s_clipped = np.clip(s, lo, hi)
    return (s_clipped - lo) / (hi - lo)

                                                              
                                                      
                                                              
def plot_attention_overlay_chunk(ax, tokens, vis_scores, x_max=0.99, title=None):

    ax.axis("off")
    if title is not None:
        ax.set_title(
            title,
            fontsize=SUBTITLE_FONTSIZE,
            fontweight=FONT_WEIGHT,
            fontfamily=FONT_FAMILY,
            pad=2
        )

    x, y = 0.01, 0.98
    line_height = 0.07
    space = 0.001

    lowest_y = y

    for tok, s in zip(tokens, vis_scores):
        tok = tok.replace("▁", " ").replace("Ġ", " ")
        if tok.strip() == "":
            continue

        rgba = (1.0, 1.0 - s, 1.0 - s, 1.0)
        token_width = 0.006 * max(len(tok), 1)

                                     
        if x + token_width > x_max:
            x = 0.01
            y -= line_height
            lowest_y = min(lowest_y, y)
            if y < 0.02:
                break

        ax.text(
            x, y, tok,
            fontsize=TOKEN_FONTSIZE,
            fontfamily=FONT_FAMILY,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=rgba,
                edgecolor="none"
            )
        )

        x += token_width + space

    ax.set_ylim(lowest_y - 0.05, 1.02)

                                                              
                            
                                                              
def plot_token_importance_ax(ax, token_attr, top_k=30):
    tokens = [t for t, _ in token_attr]
    vals = np.array([v for _, v in token_attr], dtype=float)

    k = min(top_k, len(vals))
    idx = np.argsort(np.abs(vals))[-k:][::-1]
    tokens_k = [tokens[i] for i in idx]
    vals_k = vals[idx]

    colors = ["red" if v > 0 else "blue" for v in vals_k]

    ax.barh(range(k), vals_k, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)

    ax.set_yticks(range(k))
    ax.set_yticklabels(tokens_k, fontsize=TICKLABEL_FONTSIZE, fontfamily=FONT_FAMILY)

    ax.set_xlabel(r"$\Delta \log P(\hat{y}_2)$", fontsize=AXISLABEL_FONTSIZE, fontweight=FONT_WEIGHT, fontfamily=FONT_FAMILY)
    ax.set_title(f"Token Level (Top {k})", fontsize=SUBTITLE_FONTSIZE, fontweight=FONT_WEIGHT, fontfamily=FONT_FAMILY)

    ax.tick_params(axis="x", labelsize=TICKLABEL_FONTSIZE)

                                                              
                      
                                                              
def plot_section_importance_ax(ax, section_attr, baseline_pred="yes"):
    names = list(section_attr.keys())
    vals = np.array([section_attr[n] for n in names], dtype=float)

    order = np.argsort(np.abs(vals))[::-1]
    names = [names[i] for i in order]
    vals = vals[order]

    colors = ["red" if v > 0 else "blue" for v in vals]

    ax.barh(range(len(names)), vals, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, linewidth=1)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=TICKLABEL_FONTSIZE, fontfamily=FONT_FAMILY)

    ax.set_xlabel(r"$\Delta \log P(\hat{y}_2)$", fontsize=AXISLABEL_FONTSIZE, fontweight=FONT_WEIGHT, fontfamily=FONT_FAMILY)
    plot_label = r"$\hat{y}_2$"
    
    ax.set_title(
        f"Section Level ({plot_label} = {baseline_pred.capitalize()})",
        fontsize=SUBTITLE_FONTSIZE,
        fontweight=FONT_WEIGHT,
        fontfamily=FONT_FAMILY
    )

    ax.tick_params(axis="x", labelsize=TICKLABEL_FONTSIZE)

                                                              
                                     
                                                              
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def add_attention_legend_right_edge_aligned(
    fig,
    ax_row2_right,
    y=0.64,
    h=0.22,
    w=0.055,                                                             
    pad=0.006,                            
    bar_w=0.28,                                                       
):

                                                     
    row2_x1 = ax_row2_right.get_position().x1
    x0 = row2_x1 - w - pad

    ax_leg = fig.add_axes([x0, y, w, h])
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.set_frame_on(False)

                                                  
    cmap = LinearSegmentedColormap.from_list(
        "attn_red", ["#ffffff", "#ffb3b3", "#ff0000"]
    )
    grad = np.linspace(0, 1, 256).reshape(-1, 1)

                                                        
                                                   
                                                        
    bar_x0 = 0.5 - bar_w / 2
    bar_x1 = 0.5 + bar_w / 2

    ax_leg.imshow(
        grad,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=(bar_x0, bar_x1, 0.05, 0.95),
        transform=ax_leg.transAxes,
    )

                                                        
             
                                                        
    ax_leg.text(
        0.5, 1.05,
        "Attention\n(log-scaled)",
        transform=ax_leg.transAxes,
        ha="center", va="bottom",
        fontsize=LEGEND_FONTSIZE,
        fontweight=FONT_WEIGHT,
        fontfamily=FONT_FAMILY,
    )

                                                        
                                              
                                                        
    label_x = bar_x1 + 0.06
    ax_leg.text(
        label_x, 0.95, "HIGH",
        transform=ax_leg.transAxes,
        ha="left", va="top",
        fontsize=LEGEND_FONTSIZE,
        fontweight=FONT_WEIGHT,
        fontfamily=FONT_FAMILY,
    )
    ax_leg.text(
        label_x, 0.50, "MED",
        transform=ax_leg.transAxes,
        ha="left", va="center",
        fontsize=LEGEND_FONTSIZE,
        fontweight=FONT_WEIGHT,
        fontfamily=FONT_FAMILY,
    )
    ax_leg.text(
        label_x, 0.05, "LOW",
        transform=ax_leg.transAxes,
        ha="left", va="bottom",
        fontsize=LEGEND_FONTSIZE,
        fontweight=FONT_WEIGHT,
        fontfamily=FONT_FAMILY,
    )

                                                   
    ax_leg.plot(
        [bar_x0, bar_x1, bar_x1, bar_x0, bar_x0],
        [0.05, 0.05, 0.95, 0.95, 0.05],
        transform=ax_leg.transAxes,
        linewidth=0.8,
        color="black",
    )

    return x0

                                                              
                                      
                                                              
trimmed = trim_before_marker(tokens_and_scores, SECTION_MARKERS)

raw_scores = []
tok_list = []
for tok, s in trimmed:
    tok_list.append(tok)
    if isinstance(s, list):
        raw_scores.append(float(np.mean(s)))
    else:
        raw_scores.append(float(s))

raw_scores = np.array(raw_scores, dtype=float)

attn_mass = sum_normalize(raw_scores)
attn_mass[attn_mass < ABS_SUPPRESS] = 0.0
vis_scores = quantile_rescale(log_rescale(attn_mass))

                     
mid = len(tok_list) // 2
tok_1, tok_2 = tok_list[:mid], tok_list[mid:]
vis_1, vis_2 = vis_scores[:mid], vis_scores[mid:]

                                                              
                                               
                                                              
fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[0.38, 0.38, 1.0],
    wspace=0.28                                                 
)

            
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, :])

            
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[2, 1])

                                                          
plot_token_importance_ax(ax2, token_attr, top_k=30)
plot_section_importance_ax(ax3, section_attr, baseline_pred=y_pred)

                                                   
legend_x0 = add_attention_legend_right_edge_aligned(
    fig,
    ax3,
    y=0.60,
    h=0.22,
    w=0.055,                      
    pad=0.006,
    bar_w=0.28                                    
)
                                                  
x_max_overlay = legend_x0 + 0.1

import re

def pretty_seedfold_name(seedfold_name: str) -> str:
                                            
    m = re.search(r"seed(\d+)_fold(\d+)", seedfold_name)
    if m:
        seed = m.group(1)
        fold = m.group(2)
                                     
        suffix = seedfold_name.split(f"_fold{fold}", 1)[-1]
        suffix = suffix.replace("_", " ").strip()
        if suffix:
            return f"Seed {42} | Fold {3}"
        return f"Seed {42} | Fold {3}"
    else:
                                                
        return seedfold_name.replace("_", " ")

                                  
seedfold_display = pretty_seedfold_name(seedfold_name)

                                                              
                                
                                                              

IDX = 98

subtitle_line1 = f"Clinical Summary | {seedfold_display} | Patient Index = {IDX}"
subtitle_line2 = f"Label = {str(y_true).capitalize()} | Prediction = {str(y_pred).capitalize()} | {'Correct' if correct else 'Incorrect'}"

                              
                                                 
               
                        
                                                                      
                             
                             
            
   

                                    
ax0.text(
    0.01, 1.02,
    subtitle_line1 + "\n" + subtitle_line2,
    transform=ax0.transAxes,
    fontsize=SUBTITLE_FONTSIZE,
    fontweight=FONT_WEIGHT,
    fontfamily=FONT_FAMILY,
    ha="left",
    va="bottom",
)

plot_attention_overlay_chunk(ax0, tok_1, vis_1, x_max=x_max_overlay)

ax1.text(
    0.01, 1.02,
    r"$\hat{y}_1$ & cRoT",
    transform=ax1.transAxes,
    fontsize=SUBTITLE_FONTSIZE,
    fontweight=FONT_WEIGHT,
    fontfamily=FONT_FAMILY,
    ha="left",
    va="bottom",
)

plot_attention_overlay_chunk(ax1, tok_2, vis_2, x_max=x_max_overlay)

plt.subplots_adjust(hspace=0.18, wspace=0.25, top=0.90, bottom=0.06)

save_path = os.path.join(SAVE_DIR, f"{seedfold_name}_patient{IDX}_dashboard_unified_font.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(" Saved!:", save_path)
