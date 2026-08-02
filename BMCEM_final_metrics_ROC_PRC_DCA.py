import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import numpy.core.numeric
import sys
import pickle
sys.modules['numpy._core.numeric'] = numpy.core.numeric

                        

bioclinical_bert = []

ALL_SEEDS = [7, 42, 99, 123, 2024]
N_SPLITS = 5

for seed in ALL_SEEDS:
    for fold in range(N_SPLITS):
        path = f"intermediate/bioclinical_bert/pred_seed{seed}_fold{fold}.pkl"
        
        df = pd.read_pickle(path)
        
        bioclinical_bert.append({
            "seed": seed,
            "fold": fold,
            "data": df
        })

bioclinical_bert

from sklearn.metrics import precision_score

rows = []

for item in bioclinical_bert:
    seed = item["seed"]
    fold = item["fold"]
    data = item["data"]

    y_true = data["y_true"]
    y_prob = data["y_prob"]

                                                                   
    y_pred = (y_prob >= 0.5).astype(int)

    precision = precision_score(y_true, y_pred)

    rows.append({
        "seed": seed,
        "fold": fold,
        "y_true": y_true,
        "y_prob": y_prob,
        "precision": precision,              
        **data["metrics"]
    })

metrics_df = pd.DataFrame(rows)

metrics_df

from sklearn.metrics import roc_auc_score, average_precision_score

seed_auc = {}

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    seed_auc[seed] = {
        "auroc": roc_auc_score(y_true_all, y_prob_all),
        "auprc": average_precision_score(y_true_all, y_prob_all),
    }

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)

seed_metrics = []

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true = np.concatenate(seed_df["y_true"].values)
    y_prob = np.concatenate(seed_df["y_prob"].values)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    seed_metrics.append({
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "brier": brier_score_loss(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
    })

roc_vals = [seed_auc[s]["auroc"] for s in ALL_SEEDS]
pr_vals  = [seed_auc[s]["auprc"] for s in ALL_SEEDS]

results = []

for key in seed_metrics[0].keys():
    vals = [m[key] for m in seed_metrics]

    results.append({
        "Metric": key,
        "Value": f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
    })

results_df = pd.DataFrame(results)

print("\n Results \n")
print(results_df.to_string(index=False))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

mean_fpr = np.linspace(0, 1, 200)

tprs = []
aucs = []

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    roc_auc = auc(fpr, tpr)

                                          
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)
    aucs.append(roc_auc)

                   
tprs = np.array(tprs)

mean_tpr = tprs.mean(axis=0)
std_tpr  = tprs.std(axis=0)

mean_auc = np.mean(aucs)
std_auc  = np.std(aucs, ddof=1)

      
plt.figure()

plt.plot(mean_fpr, mean_tpr,
         label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})")

plt.fill_between(mean_fpr,
                 np.maximum(mean_tpr - std_tpr, 0),
                 np.minimum(mean_tpr + std_tpr, 1),
                 alpha=0.2,
                 label="±1 std")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve")
plt.legend()
plt.grid()

plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

mean_recall = np.linspace(0, 1, 200)

precisions_interp = []
pr_aucs = []

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    pr_auc = average_precision_score(y_true_all, y_prob_all)

                                                                  
    precision = precision[::-1]
    recall = recall[::-1]

    interp_precision = np.interp(mean_recall, recall, precision)

    precisions_interp.append(interp_precision)
    pr_aucs.append(pr_auc)

precisions_interp = np.array(precisions_interp)

mean_precision = precisions_interp.mean(axis=0)
std_precision  = precisions_interp.std(axis=0)

mean_pr_auc = np.mean(pr_aucs)
std_pr_auc  = np.std(pr_aucs, ddof=1)

      
plt.figure()

plt.plot(mean_recall, mean_precision,
         label=f"Mean PRC (AUPRC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f})")

plt.fill_between(mean_recall,
                 np.maximum(mean_precision - std_precision, 0),
                 np.minimum(mean_precision + std_precision, 1),
                 alpha=0.2,
                 label="±1 std")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Mean Precision-Recall Curve")
plt.legend()
plt.grid()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

mean_thresholds = np.linspace(0, 1, 200)

sens_list = []
spec_list = []

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true = np.concatenate(seed_df["y_true"].values)
    y_prob = np.concatenate(seed_df["y_prob"].values)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    sensitivity = tpr
    specificity = 1 - fpr

                                                                
    thresholds = thresholds[::-1]
    sensitivity = sensitivity[::-1]
    specificity = specificity[::-1]

                                            
    interp_sens = np.interp(mean_thresholds, thresholds, sensitivity)
    interp_spec = np.interp(mean_thresholds, thresholds, specificity)

    sens_list.append(interp_sens)
    spec_list.append(interp_spec)

sens_list = np.array(sens_list)
spec_list = np.array(spec_list)

mean_sens = sens_list.mean(axis=0)
std_sens  = sens_list.std(axis=0)

mean_spec = spec_list.mean(axis=0)
std_spec  = spec_list.std(axis=0)

      
plt.figure()

                   
plt.plot(mean_thresholds, mean_sens, label="Sensitivity")
plt.fill_between(mean_thresholds,
                 np.maximum(mean_sens - std_sens, 0),
                 np.minimum(mean_sens + std_sens, 1),
                 alpha=0.2)

                   
plt.plot(mean_thresholds, mean_spec, label="Specificity")
plt.fill_between(mean_thresholds,
                 np.maximum(mean_spec - std_spec, 0),
                 np.minimum(mean_spec + std_spec, 1),
                 alpha=0.2)

                              
                                                           

                                       
                                                  
                                          

                                  
                                  

                                                                                               

                                 
                                              
                                              

plt.xlabel("Threshold")
plt.ylabel("Metric Value")
plt.title("Sensitivity and Specificity vs Threshold")
plt.legend()
plt.grid()

plt.show()

                    
                     
                                                                          

                                                

                       

                        
                                                      

                                                
                                                       
                                                       

                                   
                                                      

                                                                
                                                      

                                  
                                         
                                                    
           
                                                                    

                              

                                     
                                             

                        
                                                               

               
                           
                           
                                                             
                           

                               
                       
                              
                              
                              
                            
                    
        

                        
                                              

                                       
              
                                                                             
                                    
                      
                        
                                                                      
        

                                    

                                                       
                                          

                            
            

             

path = "intermediate/stage2_cRoT_all_metrics.pkl"

with open(path, "rb") as f:
    crot = pickle.load(f)

crot

rows_crot = []

for item in crot:
    row = {
        "seed": item["seed"],
        "fold": item["fold"],
        "y_true": item["y_true"],
        "y_prob": item["y_prob"],
        **item["metrics"]                      
    }
    rows_crot.append(row)

crot_metrics_df = pd.DataFrame(rows_crot)

crot_metrics_df

from sklearn.metrics import roc_auc_score, average_precision_score

crot_seed_auc = {}

for seed in ALL_SEEDS:
    seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    crot_seed_auc[seed] = {
        "auroc": roc_auc_score(y_true_all, y_prob_all),
        "auprc": average_precision_score(y_true_all, y_prob_all),
    }

crot_seed_metrics = []

for seed in ALL_SEEDS:
    seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true = np.concatenate(seed_df["y_true"].values)
    y_prob = np.concatenate(seed_df["y_prob"].values)
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    crot_seed_metrics.append({
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "brier": brier_score_loss(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
    })

roc_vals = [crot_seed_auc[s]["auroc"] for s in ALL_SEEDS]
pr_vals  = [crot_seed_auc[s]["auprc"] for s in ALL_SEEDS]

crot_results = []

for key in crot_seed_metrics[0].keys():
    vals = [m[key] for m in crot_seed_metrics]

    crot_results.append({
        "Metric": key,
        "Value": f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
    })

crot_results_df = pd.DataFrame(crot_results)

print("\n final results \n")
print(crot_results_df.to_string(index=False))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

crot_mean_fpr = np.linspace(0, 1, 200)

crot_tprs = []
crot_aucs = []

for seed in ALL_SEEDS:
    seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    crot_fpr, crot_tpr, _ = roc_curve(y_true_all, y_prob_all)
    roc_auc = auc(crot_fpr, crot_tpr)

                                          
    interp_tpr = np.interp(crot_mean_fpr, crot_fpr, crot_tpr)
    interp_tpr[0] = 0.0

    crot_tprs.append(interp_tpr)
    crot_aucs.append(roc_auc)

                   
crot_tprs = np.array(crot_tprs)

crot_mean_tpr = crot_tprs.mean(axis=0)
crot_std_tpr  = crot_tprs.std(axis=0)

crot_mean_auc = np.mean(crot_aucs)
crot_std_auc  = np.std(crot_aucs, ddof=1)

      
plt.figure()

plt.plot(crot_mean_fpr, crot_mean_tpr,
         label=f"Mean ROC (AUC = {crot_mean_auc:.4f} ± {crot_std_auc:.4f})")

plt.fill_between(crot_mean_fpr,
                 np.maximum(crot_mean_tpr - crot_std_tpr, 0),
                 np.minimum(crot_mean_tpr + crot_std_tpr, 1),
                 alpha=0.2,
                 label="±1 std")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve")
plt.legend()
plt.grid()

plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

crot_mean_recall = np.linspace(0, 1, 200)

crot_precisions_interp = []
crot_pr_aucs = []

for seed in ALL_SEEDS:
    crot_seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true_all = np.concatenate(crot_seed_df["y_true"].values)
    y_prob_all = np.concatenate(crot_seed_df["y_prob"].values)

    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    crot_pr_auc = average_precision_score(y_true_all, y_prob_all)

                                                                  
    precision = precision[::-1]
    recall = recall[::-1]

    crot_interp_precision = np.interp(crot_mean_recall, recall, precision)

    crot_precisions_interp.append(crot_interp_precision)
    crot_pr_aucs.append(crot_pr_auc)

crot_precisions_interp = np.array(crot_precisions_interp)

crot_mean_precision = crot_precisions_interp.mean(axis=0)
crot_std_precision  = crot_precisions_interp.std(axis=0)

crot_mean_pr_auc = np.mean(crot_pr_aucs)
crot_std_pr_auc  = np.std(crot_pr_aucs, ddof=1)

      
plt.figure()

plt.plot(crot_mean_recall, crot_mean_precision,
         label=f"Mean PRC (AUPRC = {crot_mean_pr_auc:.4f} ± {crot_std_pr_auc:.4f})")

plt.fill_between(crot_mean_recall,
                 np.maximum(crot_mean_precision - crot_std_precision, 0),
                 np.minimum(crot_mean_precision + crot_std_precision, 1),
                 alpha=0.2,
                 label="±1 std")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Mean Precision-Recall Curve")
plt.legend()
plt.grid()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

crot_mean_thresholds = np.linspace(0, 1, 200)

crot_sens_list = []
crot_spec_list = []

for seed in ALL_SEEDS:
    crot_seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true = np.concatenate(crot_seed_df["y_true"].values)
    y_prob = np.concatenate(crot_seed_df["y_prob"].values)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    sensitivity = tpr
    specificity = 1 - fpr

                                                                
    thresholds = thresholds[::-1]
    sensitivity = sensitivity[::-1]
    specificity = specificity[::-1]

                                            
    interp_sens = np.interp(crot_mean_thresholds, thresholds, sensitivity)
    interp_spec = np.interp(crot_mean_thresholds, thresholds, specificity)

    crot_sens_list.append(interp_sens)
    crot_spec_list.append(interp_spec)

crot_sens_list = np.array(crot_sens_list)
crot_spec_list = np.array(crot_spec_list)

crot_mean_sens = crot_sens_list.mean(axis=0)
crot_std_sens  = crot_sens_list.std(axis=0)

crot_mean_spec = crot_spec_list.mean(axis=0)
crot_std_spec  = crot_spec_list.std(axis=0)

      
plt.figure()

                   
plt.plot(crot_mean_thresholds, crot_mean_sens, label="Sensitivity")
plt.fill_between(crot_mean_thresholds,
                 np.maximum(crot_mean_sens - crot_std_sens, 0),
                 np.minimum(crot_mean_sens + crot_std_sens, 1),
                 alpha=0.2)

                   
plt.plot(crot_mean_thresholds, crot_mean_spec, label="Specificity")
plt.fill_between(crot_mean_thresholds,
                 np.maximum(crot_mean_spec - crot_std_spec, 0),
                 np.minimum(crot_mean_spec + crot_std_spec, 1),
                 alpha=0.2)

                              
                                                           

                                       
                                                  
                                          

                                  
                                  

                                                                                               

                                 
                                              
                                              

plt.xlabel("Threshold")
plt.ylabel("Metric Value")
plt.title("Sensitivity and Specificity vs Threshold")
plt.legend()
plt.grid()

plt.show()

                                              

def evaluate_model(metrics_df, target_sens_list=[0.90, 0.95]):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        roc_auc_score, average_precision_score,
        confusion_matrix, brier_score_loss, roc_curve
    )

    ALL_SEEDS = sorted(metrics_df["seed"].unique())

    seed_metrics = []
    highsens_metrics = []

    for seed in ALL_SEEDS:
        seed_df = metrics_df[metrics_df["seed"] == seed]

                                  
        y_true = np.concatenate(seed_df["y_true"].values)
        y_prob = np.concatenate(seed_df["y_prob"].values)

                                   
                                     
                                   
        y_pred_05 = (y_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_05).ravel()

        seed_metrics.append({
            "accuracy": accuracy_score(y_true, y_pred_05),
            "f1": f1_score(y_true, y_pred_05),
            "precision": precision_score(y_true, y_pred_05, zero_division=0),
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "brier": brier_score_loss(y_true, y_prob),
            "auroc": roc_auc_score(y_true, y_prob),
            "auprc": average_precision_score(y_true, y_prob),
        })

                                   
                                                         
                                   
        for target_sens in target_sens_list:
        
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
            idx_candidates = np.where(tpr >= target_sens)[0]
            if len(idx_candidates) == 0:
                idx = np.argmin(np.abs(tpr - target_sens))
            else:
                idx = idx_candidates[0]
        
            thresh = thresholds[idx]
            y_pred_hs = (y_prob >= thresh).astype(int)
        
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_hs).ravel()
        
            highsens_metrics.append({
                "target_sens": target_sens,
                "threshold": thresh,
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "precision": precision_score(y_true, y_pred_hs, zero_division=0),
                "npv": tn / (tn + fn),
            })

                               
                       
                               

                              
    results = []
    for key in seed_metrics[0].keys():
        vals = [m[key] for m in seed_metrics]
        results.append({
            "Metric": key,
            "Value": f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
        })

    results_df = pd.DataFrame(results)

                                                                      
    hs_df = pd.DataFrame(highsens_metrics)
    
    hs_results = []
    
    for target_sens in hs_df["target_sens"].unique():
        subset = hs_df[hs_df["target_sens"] == target_sens]
    
        for key in ["threshold", "sensitivity", "specificity", "precision", "npv"]:
            vals = subset[key].values
    
            hs_results.append({
                "Target Sensitivity": f"{int(target_sens*100)}%",
                "Metric": key,
                "Value": f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
            })
    
    highsens_df = pd.DataFrame(hs_results)

    return results_df, highsens_df

res1, hs1 = evaluate_model(metrics_df)
res2, hs2 = evaluate_model(crot_metrics_df)

print("\nBioclinical BERT:\n", res1)
print("\nBioclinical BERT (High Sens):\n", hs1)

print("\ncRoT:\n", res2)
print("\ncRoT (High Sens):\n", hs2)

                    

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

                 
mean_fpr = np.linspace(0, 1, 200)

                           
                   
                           
tprs = []
aucs = []

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    roc_auc = auc(fpr, tpr)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)
    aucs.append(roc_auc)

tprs = np.array(tprs)
mean_tpr = tprs.mean(axis=0)
std_tpr  = tprs.std(axis=0)

mean_auc = np.mean(aucs)
std_auc  = np.std(aucs, ddof=1)

                           
                
                           
crot_tprs = []
crot_aucs = []

for seed in ALL_SEEDS:
    seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    roc_auc = auc(fpr, tpr)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    crot_tprs.append(interp_tpr)
    crot_aucs.append(roc_auc)

crot_tprs = np.array(crot_tprs)
crot_mean_tpr = crot_tprs.mean(axis=0)
crot_std_tpr  = crot_tprs.std(axis=0)

crot_mean_auc = np.mean(crot_aucs)
crot_std_auc  = np.std(crot_aucs, ddof=1)

                           
             
                           
plt.figure(figsize=(10, 10))

              
plt.plot(mean_fpr, crot_mean_tpr,
         label=f"qLoRA (cRoT) (AUROC = {crot_mean_auc:.4f} ± {crot_std_auc:.4f})")

plt.fill_between(mean_fpr,
                 np.maximum(crot_mean_tpr - crot_std_tpr, 0),
                 np.minimum(crot_mean_tpr + crot_std_tpr, 1),
                 alpha=0.2)

                 
plt.plot(mean_fpr, mean_tpr,
         label=f"BioClinical BERT (AUROC = {mean_auc:.4f} ± {std_auc:.4f})")

plt.fill_between(mean_fpr,
                 np.maximum(mean_tpr - std_tpr, 0),
                 np.minimum(mean_tpr + std_tpr, 1),
                 alpha=0.2)

          
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristics")
plt.legend()
plt.grid()
plt.savefig("BMCEM_curves/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

                     

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

                    
mean_recall = np.linspace(0, 1, 200)

                           
                   
                           
precisions = []
pr_aucs = []

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    pr_auc = average_precision_score(y_true_all, y_prob_all)

                               
    precision = precision[::-1]
    recall = recall[::-1]

    interp_precision = np.interp(mean_recall, recall, precision)

    precisions.append(interp_precision)
    pr_aucs.append(pr_auc)

precisions = np.array(precisions)
mean_precision = precisions.mean(axis=0)
std_precision  = precisions.std(axis=0)

mean_pr_auc = np.mean(pr_aucs)
std_pr_auc  = np.std(pr_aucs, ddof=1)

                           
                
                           
crot_precisions = []
crot_pr_aucs = []

for seed in ALL_SEEDS:
    seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true_all = np.concatenate(seed_df["y_true"].values)
    y_prob_all = np.concatenate(seed_df["y_prob"].values)

    precision, recall, _ = precision_recall_curve(y_true_all, y_prob_all)
    pr_auc = average_precision_score(y_true_all, y_prob_all)

    precision = precision[::-1]
    recall = recall[::-1]

    interp_precision = np.interp(mean_recall, recall, precision)

    crot_precisions.append(interp_precision)
    crot_pr_aucs.append(pr_auc)

crot_precisions = np.array(crot_precisions)
crot_mean_precision = crot_precisions.mean(axis=0)
crot_std_precision  = crot_precisions.std(axis=0)

crot_mean_pr_auc = np.mean(crot_pr_aucs)
crot_std_pr_auc  = np.std(crot_pr_aucs, ddof=1)

                           
             
                           
plt.figure(figsize=(10, 10))

              
plt.plot(mean_recall, crot_mean_precision,
         label=f"qLoRA (cRoT) (AUPRC = {crot_mean_pr_auc:.4f} ± {crot_std_pr_auc:.4f})")

plt.fill_between(mean_recall,
                 np.maximum(crot_mean_precision - crot_std_precision, 0),
                 np.minimum(crot_mean_precision + crot_std_precision, 1),
                 alpha=0.2)

                 
plt.plot(mean_recall, mean_precision,
         label=f"BioClinical BERT (AUPRC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f})")

plt.fill_between(mean_recall,
                 np.maximum(mean_precision - std_precision, 0),
                 np.minimum(mean_precision + std_precision, 1),
                 alpha=0.2)

                       
prevalence = np.mean(y_true_all)
plt.hlines(prevalence, 0, 1, linestyles="--", label=f"Baseline prevalence = {prevalence:.4f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend(loc="lower left", bbox_to_anchor=(0.05, 0.08))

plt.grid()

plt.savefig("BMCEM_curves/prc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

import numpy as np
from sklearn.metrics import roc_auc_score

                           
                                         
                           
def get_pooled_predictions(metrics_df, seed):
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true = np.concatenate(seed_df["y_true"].values)
    y_prob = np.concatenate(seed_df["y_prob"].values)

    return y_true, y_prob

                           
                                      
                           
def bootstrap_delong(y_true, pred1, pred2, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []

    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))

                                 
        if len(np.unique(y_true[idx])) < 2:
            continue

        auc1 = roc_auc_score(y_true[idx], pred1[idx])
        auc2 = roc_auc_score(y_true[idx], pred2[idx])

        diffs.append(auc2 - auc1)

    diffs = np.array(diffs)

                                              
    p_value = np.mean(diffs <= 0)

    return p_value

                           
                          
                           
p_values = []
auc_diffs = []

for seed in ALL_SEEDS:
                              
    y_true_1, y_prob_1 = get_pooled_predictions(metrics_df, seed)

                           
    y_true_2, y_prob_2 = get_pooled_predictions(crot_metrics_df, seed)

                                   
    assert np.array_equal(y_true_1, y_true_2), f"Mismatch in seed {seed}"

                      
    auc1 = roc_auc_score(y_true_1, y_prob_1)
    auc2 = roc_auc_score(y_true_1, y_prob_2)

    auc_diffs.append(auc2 - auc1)

                        
    p = bootstrap_delong(y_true_1, y_prob_1, y_prob_2)
    p_values.append(p)

                           
                   
                           
mean_diff = np.mean(auc_diffs)
std_diff  = np.std(auc_diffs, ddof=1)

mean_p = np.mean(p_values)
std_p  = np.std(p_values, ddof=1)

print("\n Test of significance (per-seed aggregation)\n")
print(f"ΔAUROC (qLoRA (cRoT) - BioClinical BERT): {mean_diff:.4f} ± {std_diff:.4f}")
print(f"p-value: {mean_p:.6f} ± {std_p:.6f}")
print(f"Per-seed p-values: {p_values}")

from scipy.stats import combine_pvalues

                                        
stat, p_fisher = combine_pvalues(p_values, method='fisher')

print("\n Combined p-values")
print(f"Fisher's method p-value: {p_fisher:.6e}")

                                                                                                                      
import numpy as np
import matplotlib.pyplot as plt

                         
thresholds = np.linspace(0.01, 0.99, 100)

def compute_net_benefit(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))

        nb = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(nb)

    return np.array(net_benefits)

                           
                   
                           
nb_bert_all = []

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true = np.concatenate(seed_df["y_true"].values)
    y_prob = np.concatenate(seed_df["y_prob"].values)

    nb = compute_net_benefit(y_true, y_prob, thresholds)
    nb_bert_all.append(nb)

nb_bert_all = np.array(nb_bert_all)
mean_nb_bert = nb_bert_all.mean(axis=0)
std_nb_bert  = nb_bert_all.std(axis=0)

                           
                
                           
nb_crot_all = []

for seed in ALL_SEEDS:
    seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true = np.concatenate(seed_df["y_true"].values)
    y_prob = np.concatenate(seed_df["y_prob"].values)

    nb = compute_net_benefit(y_true, y_prob, thresholds)
    nb_crot_all.append(nb)

nb_crot_all = np.array(nb_crot_all)
mean_nb_crot = nb_crot_all.mean(axis=0)
std_nb_crot  = nb_crot_all.std(axis=0)

                           
           
                           
           
prevalence = np.mean(y_true)
treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

            
treat_none = np.zeros_like(thresholds)

                           
        
                           
plt.figure()
plt.xlim(0, 1)
plt.ylim(-0.05, 0.8)
       
plt.plot(thresholds, mean_nb_crot,
         label="qLoRA (cRoT)")
plt.fill_between(thresholds,
                 mean_nb_crot - std_nb_crot,
                 mean_nb_crot + std_nb_crot,
                 alpha=0.2)

                 
plt.plot(thresholds, mean_nb_bert,
         label="BioClinical BERT")
plt.fill_between(thresholds,
                 mean_nb_bert - std_nb_bert,
                 mean_nb_bert + std_nb_bert,
                 alpha=0.2)

           
plt.plot(thresholds, treat_all, linestyle="--", label="Consider all as true stroke")
plt.plot(thresholds, treat_none, linestyle="--", label="Consider all as stroke mimic")

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis")
plt.legend()
plt.grid()

plt.show()

                           
        
                           
plt.figure()
plt.xlim(0, 1)
plt.ylim(-0.05, 0.8)

             
plt.plot(
    thresholds,
    mean_nb_crot,
    color="red",
    label="qLoRA (cRoT)"
)
plt.fill_between(
    thresholds,
    mean_nb_crot - std_nb_crot,
    mean_nb_crot + std_nb_crot,
    color="red",
    alpha=0.2
)

                        
plt.plot(
    thresholds,
    mean_nb_bert,
    color="blue",
    label="BioClinical BERT"
)
plt.fill_between(
    thresholds,
    mean_nb_bert - std_nb_bert,
    mean_nb_bert + std_nb_bert,
    color="blue",
    alpha=0.2
)

                       
plt.plot(thresholds, treat_all, linestyle="--", color="black", label="Consider all as true stroke")
plt.plot(thresholds, treat_none, linestyle="--", color="dimgray", label="Consider all as stroke mimic")

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("DCA.png", dpi=300, bbox_inches="tight")
plt.show()

                                         
idx = np.where(mean_nb_crot > treat_all)[0]

if len(idx) > 0:
    first_idx = idx[0]
    print(f"qLoRA first exceeds Treat All at:")
    print(f"  Threshold = {thresholds[first_idx]:.3f}")
    print(f"  qLoRA NB  = {mean_nb_crot[first_idx]:.4f}")
    print(f"  Treat All = {treat_all[first_idx]:.4f}")
else:
    print("qLoRA never exceeds Treat All.")

                                              
mask = mean_nb_crot > treat_all

if np.any(mask):
    start = thresholds[mask][0]
    end = thresholds[mask][-1]

    print(f"qLoRA exceeds the Treat All strategy for threshold probabilities "
          f"from {start:.2f} to {end:.2f}.")
else:
    print("qLoRA never exceeds the Treat All strategy.")

diff = mean_nb_crot - treat_all
mask = diff > 0

changes = np.where(np.diff(mask.astype(int)) != 0)[0]

print("Crossings near thresholds:")
for i in changes:
    print(f"{thresholds[i]:.4f} -> {thresholds[i+1]:.4f}")

             

           
               

                                        

                             
                             
                             
                                                                                  
                                   

                           

                             
                   
                             
                                                                              
                               

                               
                               

                             
                  
                             
                                                                             
                              

                                          
                                    

                             
                  
                             
                                                                                 
                                  

                             
                                           
                             
                                                                                     
                                

                                  
                                  

                                          

                    
                     

                                
                    

                             
                
                 
                                            

                        
                                        
                                

                          

                                                                  

                                                  

               

from sklearn.metrics import matthews_corrcoef

                           
                    
                           
crot_mcc_list = []

for seed in ALL_SEEDS:
    crot_seed_df = crot_metrics_df[crot_metrics_df["seed"] == seed]

    y_true = np.concatenate(crot_seed_df["y_true"].values)
    y_prob = np.concatenate(crot_seed_df["y_prob"].values)

    mccs = []
    for t in crot_mean_thresholds:
        y_pred = (y_prob >= t).astype(int)

                                
        if len(np.unique(y_pred)) < 2:
            mccs.append(0)
        else:
            mccs.append(matthews_corrcoef(y_true, y_pred))

    crot_mcc_list.append(mccs)

crot_mcc_list = np.array(crot_mcc_list)
crot_mean_mcc = crot_mcc_list.mean(axis=0)
crot_std_mcc  = crot_mcc_list.std(axis=0)

                           
                       
                           
mcc_list = []

for seed in ALL_SEEDS:
    seed_df = metrics_df[metrics_df["seed"] == seed]

    y_true = np.concatenate(seed_df["y_true"].values)
    y_prob = np.concatenate(seed_df["y_prob"].values)

    mccs = []
    for t in mean_thresholds:
        y_pred = (y_prob >= t).astype(int)

        if len(np.unique(y_pred)) < 2:
            mccs.append(0)
        else:
            mccs.append(matthews_corrcoef(y_true, y_pred))

    mcc_list.append(mccs)

mcc_list = np.array(mcc_list)
mean_mcc = mcc_list.mean(axis=0)
std_mcc  = mcc_list.std(axis=0)

plt.figure(figsize=(10, 10))

                
plt.plot(mean_recall, crot_mean_precision,
         color="red",
         label=f"qLoRA (cRoT) (AUPRC = {crot_mean_pr_auc:.4f} ± {crot_std_pr_auc:.4f})")

plt.fill_between(mean_recall,
                 np.maximum(crot_mean_precision - crot_std_precision, 0),
                 np.minimum(crot_mean_precision + crot_std_precision, 1),
                 color="red",
                 alpha=0.2)

                   
plt.plot(mean_recall, mean_precision,
         color="blue",
         label=f"BioClinical BERT (AUPRC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f})")

plt.fill_between(mean_recall,
                 np.maximum(mean_precision - std_precision, 0),
                 np.minimum(mean_precision + std_precision, 1),
                 color="blue",
                 alpha=0.2)

                       
prevalence = np.mean(y_true_all)
plt.hlines(prevalence, 0, 1,
           linestyles="--",
           color="black",
           label=f"Baseline prevalence = {prevalence:.4f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")

plt.legend(loc="lower left", bbox_to_anchor=(0.05, 0.08))
plt.grid(alpha = 0.2)

plt.savefig("BMCEM_curves/prc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 10))

                
plt.plot(mean_fpr, crot_mean_tpr,
         color="red",
         label=f"qLoRA (cRoT) (AUROC = {crot_mean_auc:.4f} ± {crot_std_auc:.4f})")

plt.fill_between(mean_fpr,
                 np.maximum(crot_mean_tpr - crot_std_tpr, 0),
                 np.minimum(crot_mean_tpr + crot_std_tpr, 1),
                 color="red",
                 alpha=0.2)

                   
plt.plot(mean_fpr, mean_tpr,
         color="blue",
         label=f"BioClinical BERT (AUROC = {mean_auc:.4f} ± {std_auc:.4f})")

plt.fill_between(mean_fpr,
                 np.maximum(mean_tpr - std_tpr, 0),
                 np.minimum(mean_tpr + std_tpr, 1),
                 color="blue",
                 alpha=0.2)

                              
plt.plot([0, 1], [0, 1], linestyle="--", color="black")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristics")

plt.legend()
plt.grid(alpha = 0.2)

plt.savefig("BMCEM_curves/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

                                       

import matplotlib.pyplot as plt

                         
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

                           
                      
                           
ax = axes[0]

ax.plot(crot_mean_thresholds, crot_mean_sens, label="Recall", color="red")
ax.fill_between(crot_mean_thresholds,
                np.maximum(crot_mean_sens - crot_std_sens, 0),
                np.minimum(crot_mean_sens + crot_std_sens, 1),
                color="red", alpha=0.2)

ax.plot(crot_mean_thresholds, crot_mean_spec, label="Specificity", color="blue")
ax.fill_between(crot_mean_thresholds,
                np.maximum(crot_mean_spec - crot_std_spec, 0),
                np.minimum(crot_mean_spec + crot_std_spec, 1),
                color="blue", alpha=0.2)

ax.plot(crot_mean_thresholds, crot_mean_mcc,
        label="MCC", color="green")

ax.fill_between(crot_mean_thresholds,
                np.maximum(crot_mean_mcc - crot_std_mcc, -1),
                np.minimum(crot_mean_mcc + crot_std_mcc, 1),
                color="green", alpha=0.2)

idx_90 = np.where(crot_mean_sens >= 0.90)[0][-1]
thresh_90 = crot_mean_thresholds[idx_90]

axes[0].axvline(thresh_90, linestyle=":", color="black", label=f"90% sensitivity threshold : {thresh_90:.4f}")

                         
idx_95 = np.where(crot_mean_sens >= 0.95)[0][-1]
thresh_95 = crot_mean_thresholds[idx_95]

axes[0].axvline(thresh_95,
                linestyle="--",
                color="black",
                label=f"95% sensitivity threshold : {thresh_95:.4f}")

ax.set_title("qLoRA (cRoT)")
ax.set_xlabel("Threshold")
ax.set_ylabel("Metric")
ax.grid(alpha = 0.2)
ax.legend(loc="lower right")

                           
                          
                           
ax = axes[1]

ax.plot(mean_thresholds, mean_sens, label="Recall", color="red")
ax.fill_between(mean_thresholds,
                np.maximum(mean_sens - std_sens, 0),
                np.minimum(mean_sens + std_sens, 1),
                color="red", alpha=0.2)

ax.plot(mean_thresholds, mean_spec, label="Specificity", color="blue")
ax.fill_between(mean_thresholds,
                np.maximum(mean_spec - std_spec, 0),
                np.minimum(mean_spec + std_spec, 1),
                color="blue", alpha=0.2)

ax.plot(mean_thresholds, mean_mcc,
        label="MCC", color="green")

ax.fill_between(mean_thresholds,
                np.maximum(mean_mcc - std_mcc, -1),
                np.minimum(mean_mcc + std_mcc, 1),
                color="green", alpha=0.2)

idx_90 = np.where(mean_sens >= 0.90)[0][-1]
thresh_90 = mean_thresholds[idx_90]

axes[1].axvline(thresh_90, linestyle=":", color="black", label=f"90% sensitivity threshold : {thresh_90:.4f}")

                         
idx_95 = np.where(mean_sens >= 0.95)[0][-1]
thresh_95 = mean_thresholds[idx_95]

axes[1].axvline(thresh_95,
                linestyle="--",
                color="black",
                label=f"95% sensitivity threshold : {thresh_95:.4f}")

ax.set_title("BioClinical BERT")
ax.set_xlabel("Threshold")
ax.grid(alpha = 0.2)
ax.legend(loc="lower right")

                           
                  
                           
plt.tight_layout()

plt.savefig("BMCEM_curves/sensitivity_threshold_plot.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

res1, hs1 = evaluate_model(metrics_df)
res2, hs2 = evaluate_model(crot_metrics_df)

print("\nBioclinical BERT:\n", res1)
print("\nBioclinical BERT (High Sens):\n", hs1)

print("\ncRoT:\n", res2)
print("\ncRoT (High Sens):\n", hs2)

import pandas as pd

output_path = "BMCEM_curves/model_results.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    
                     
    res1.to_excel(writer, sheet_name="BERT_Overall", index=False)
    hs1.to_excel(writer, sheet_name="BERT_HighSens", index=False)
    
                  
    res2.to_excel(writer, sheet_name="cRoT_Overall", index=False)
    hs2.to_excel(writer, sheet_name="cRoT_HighSens", index=False)

print(f" Saved to {output_path}")

import matplotlib.pyplot as plt

                           
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                           
             
                           
ax = axes[0]

       
ax.plot(mean_fpr, crot_mean_tpr,
        color="red",
        label=f"qLoRA (cRoT) (AUROC = {crot_mean_auc:.4f} ± {crot_std_auc:.4f})")

ax.fill_between(mean_fpr,
                np.maximum(crot_mean_tpr - crot_std_tpr, 0),
                np.minimum(crot_mean_tpr + crot_std_tpr, 1),
                color="red",
                alpha=0.2)

                 
ax.plot(mean_fpr, mean_tpr,
        color="blue",
        label=f"BioClinical BERT (AUROC = {mean_auc:.4f} ± {std_auc:.4f})")

ax.fill_between(mean_fpr,
                np.maximum(mean_tpr - std_tpr, 0),
                np.minimum(mean_tpr + std_tpr, 1),
                color="blue",
                alpha=0.2)

          
ax.plot([0, 1], [0, 1], linestyle="--", color="black")

ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Recall")
ax.set_title("Receiver Operating Characteristics")
ax.grid(alpha=0.2)
ax.legend(loc ="lower right", bbox_to_anchor=(0.95, 0.12))

                           
              
                           
ax = axes[1]

       
ax.plot(mean_recall, crot_mean_precision,
        color="red",
        label=f"qLoRA (cRoT) (AUPRC = {crot_mean_pr_auc:.4f} ± {crot_std_pr_auc:.4f})")

ax.fill_between(mean_recall,
                np.maximum(crot_mean_precision - crot_std_precision, 0),
                np.minimum(crot_mean_precision + crot_std_precision, 1),
                color="red",
                alpha=0.2)

                 
ax.plot(mean_recall, mean_precision,
        color="blue",
        label=f"BioClinical BERT (AUPRC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f})")

ax.fill_between(mean_recall,
                np.maximum(mean_precision - std_precision, 0),
                np.minimum(mean_precision + std_precision, 1),
                color="blue",
                alpha=0.2)

          
prevalence = np.mean(y_true_all)
ax.hlines(prevalence, 0, 1,
          linestyles="--",
          color="black",
          label=f"Baseline prevalence = {prevalence:.4f}")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision Recall Characteristics")
ax.grid(alpha=0.2)
ax.legend(loc="lower left", bbox_to_anchor=(0.05, 0.08))

                           
                  
                           
plt.tight_layout()

plt.savefig("BMCEM_curves/roc_prc_combined.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white")
plt.show()

print("Hooray!")
