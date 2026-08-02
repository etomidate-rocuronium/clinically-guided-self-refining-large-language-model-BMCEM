import os
import gc
import json
import random
import shutil
import warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from tqdm.auto import tqdm
from scipy.special import logit, expit
from scipy.optimize import brentq

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

import sys
import numpy as np

try:
    import numpy._core.numeric
except ImportError:
    import numpy.core.numeric
    import numpy.core.multiarray
    import numpy.core.umath

    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.umath"] = numpy.core.umath

warnings.filterwarnings("ignore")

ALL_SEEDS = [7, 42, 99, 123, 2024]
N_SPLITS = 5

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

BASE_DIR_STAGE1 = Path("<STAGE1_MODEL_DIR>")
BASE_DIR_STAGE2 = Path("<STAGE2_MODEL_DIR>")
INTERMEDIATE_DIR = Path("<INTERMEDIATE_DATA_DIR>")
RESULTS_DIR = Path("<RESULTS_DIR>")
RANK_RESULTS_DIR = Path("<RANK_RESULTS_DIR>")

N_CALIBRATION_BINS = 10
CALIBRATION_BIN_STRATEGY = "quantile"
PLATT_CV_SPLITS = 5

PATIENT_ID_COL = None

RESUME_COMPLETED_JOBS = True

MAX_INPUT_LENGTH = 4096

FINAL_SCALER_FILENAME = "stage2_final_platt_scaler.joblib"

def initialize_distributed():
    required = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    missing = [name for name in required if name not in os.environ]

    if missing:
        raise RuntimeError(
            "This script must be launched with torchrun. Missing environment "
            f"variables: {missing}\n"
            "Example: torchrun --standalone --nproc_per_node=1 "
            "posthoc_calibration_torchrun.py"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this torchrun job. Launch torchrun from "
            "a GPU-allocated node/session."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=2),
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device

RANK, WORLD_SIZE, LOCAL_RANK, DEVICE = initialize_distributed()
IS_MAIN = RANK == 0

def rank_print(*args, **kwargs):
    print(f"[rank {RANK}/{WORLD_SIZE}]", *args, **kwargs, flush=True)

rank_print(
    f"Using {DEVICE}: {torch.cuda.get_device_name(LOCAL_RANK)}"
)

HF_TOKEN = "<REDACTED_SECRET>"

if IS_MAIN:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RANK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

dist.barrier()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    token=HF_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def stage1_prompt(row):
    return f"""
You are a neurologist working in the emergency department.
Decide if the patient has acute stroke.
Answer ONLY 'yes' or 'no'.

Patient summary: {row['translated_summary_cleaned']}
label:
""".strip()

def stage2_prompt(row):
    return f"""
You are a neurologist working in the emergency department.
Given the patient summary below and the previous prediction '{row['pred_stage1']}', please reconsider and provide the correct diagnosis ('yes' or 'no').

Patient summary: {row['translated_summary_cleaned']}
Previous prediction: {row['pred_stage1']}

Note the following rule of thumb.
1. The more the number of neurological deficits present, the likelier of true stroke.
2. Presence of hypertension, diabetes mellitus, dyslipidemia increases the likelihood of true stroke.
3. Presence of cancer increases the likelihood of true stroke.
4. Presence of unilateral side weakness increases the likelihood of true stroke.
5. History of epilepsy or seizure decreases the likelihood of true stroke.
6. Presence of symmetric weakness decreases the likelihood of true stroke.

label:
""".strip()

def collect_single_token_ids(candidates):
    token_ids = set()
    for candidate in candidates:
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if len(ids) == 1:
            token_ids.add(ids[0])
    return sorted(token_ids)

YES_TOKEN_IDS = collect_single_token_ids(
    ["yes", " yes", "\nyes", "Yes", " Yes", "\nYes"]
)
NO_TOKEN_IDS = collect_single_token_ids(
    ["no", " no", "\nno", "No", " No", "\nNo"]
)

if not YES_TOKEN_IDS or not NO_TOKEN_IDS:
    raise RuntimeError(
        f"Could not identify single-token yes/no variants. "
        f"yes={YES_TOKEN_IDS}, no={NO_TOKEN_IDS}"
    )
if set(YES_TOKEN_IDS).intersection(NO_TOKEN_IDS):
    raise RuntimeError("Overlapping yes/no token IDs detected.")

rank_print(f"YES token IDs={YES_TOKEN_IDS}; NO token IDs={NO_TOKEN_IDS}")

def normalize_labels(series):
    mapping = {
        1: "yes",
        0: "no",
        True: "yes",
        False: "no",
        "1": "yes",
        "0": "no",
        "yes": "yes",
        "no": "no",
        "Yes": "yes",
        "No": "no",
        "YES": "yes",
        "NO": "no",
    }
    normalized = series.map(mapping)
    if normalized.isna().any():
        invalid = series[normalized.isna()].unique()
        raise ValueError(f"Unsupported outcome values: {invalid}")
    return normalized

def detect_patient_id_column(df):
    if PATIENT_ID_COL is not None:
        if PATIENT_ID_COL not in df.columns:
            raise KeyError(
                f"PATIENT_ID_COL='{PATIENT_ID_COL}' is absent from the pickle."
            )
        return PATIENT_ID_COL

    candidates = [
        "patient_id",
        "visit_id",
        "encounter_id",
        "subject_id",
        "case_id",
        "record_id",
        "id",
    ]
    for column in candidates:
        if column in df.columns:
            return column
    return None

def prepare_validation_dataframe(df, seed, fold):
    df = df.copy()
    required = ["translated_summary_cleaned", "y"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["y"] = normalize_labels(df["y"])
    id_column = detect_patient_id_column(df)

    if id_column is not None:
        df["patient_id"] = df[id_column].astype(str)
    else:

        hash_source = (
            df["translated_summary_cleaned"].astype(str)
            + "||"
            + df["y"].astype(str)
        )
        df["patient_id"] = pd.util.hash_pandas_object(
            hash_source, index=False
        ).astype(str)
        rank_print(
            "WARNING: no patient ID column found; using a summary+label hash. "
            "Set PATIENT_ID_COL to a stable identifier for manuscript analysis."
        )

    df["seed"] = seed
    df["fold"] = fold
    return df.reset_index(drop=True)

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": LOCAL_RANK},
        token=HF_TOKEN,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = True
    model.eval()
    return model

def load_adapter(adapter_path):
    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    base_model = load_base_model()
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    return model

def aggregate_token_logits(next_token_logits, token_ids):
    token_id_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=next_token_logits.device,
    )
    return torch.logsumexp(next_token_logits[token_id_tensor], dim=0)

def get_yes_no_probability(next_token_logits):
    yes_logit = aggregate_token_logits(next_token_logits, YES_TOKEN_IDS)
    no_logit = aggregate_token_logits(next_token_logits, NO_TOKEN_IDS)
    binary_logits = torch.stack([no_logit, yes_logit])
    binary_probabilities = torch.softmax(binary_logits, dim=0)

    probability_no = float(binary_probabilities[0].item())
    probability_yes = float(binary_probabilities[1].item())
    raw_logit = float((yes_logit - no_logit).item())
    return probability_yes, probability_no, raw_logit

def tokenize_prompt(prompt):
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(DEVICE)

def generate_stage1_predictions(df, model, seed, fold):
    predictions = []
    probabilities = []

    iterator = tqdm(
        range(len(df)),
        desc=f"rank{RANK} stage1 s{seed} f{fold}",
        position=RANK,
        leave=False,
    )

    for row_number in iterator:
        row = df.iloc[row_number]
        inputs = tokenize_prompt(stage1_prompt(row))
        with torch.inference_mode():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[0, -1, :]
        probability_yes, probability_no, _ = get_yes_no_probability(
            next_token_logits
        )
        predictions.append("yes" if probability_yes >= probability_no else "no")
        probabilities.append(probability_yes)

        del inputs, outputs, next_token_logits

    output_df = df.copy()
    output_df["pred_stage1"] = predictions
    output_df["prob_stage1"] = probabilities
    return output_df

def generate_stage2_predictions(df, model, seed, fold):
    probabilities = []
    raw_logits = []
    predictions = []

    iterator = tqdm(
        range(len(df)),
        desc=f"rank{RANK} stage2 s{seed} f{fold}",
        position=RANK,
        leave=False,
    )

    for row_number in iterator:
        row = df.iloc[row_number]
        inputs = tokenize_prompt(stage2_prompt(row))
        with torch.inference_mode():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[0, -1, :]
        probability_yes, probability_no, raw_logit = get_yes_no_probability(
            next_token_logits
        )

        probabilities.append(probability_yes)
        raw_logits.append(raw_logit)
        predictions.append(int(probability_yes >= probability_no))

        del inputs, outputs, next_token_logits

    y_true = df["y"].map({"yes": 1, "no": 0}).to_numpy(dtype=int)

    return pd.DataFrame(
        {
            "patient_id": df["patient_id"].astype(str),
            "seed": df["seed"].to_numpy(),
            "fold": df["fold"].to_numpy(),
            "y_true": y_true,
            "y_prob_raw": probabilities,
            "y_pred_raw": predictions,
            "raw_binary_logit": raw_logits,
            "pred_stage1": df["pred_stage1"].to_numpy(),
            "prob_stage1": df["prob_stage1"].to_numpy(),
        }
    )

def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

PREDICTION_COLUMNS = [
    "patient_id",
    "seed",
    "fold",
    "y_true",
    "y_prob_raw",
    "y_pred_raw",
    "raw_binary_logit",
    "pred_stage1",
    "prob_stage1",
]

def job_prediction_path(seed, fold):
    return RANK_RESULTS_DIR / f"oof_seed{seed}_fold{fold}.csv"

def run_inference_jobs():
    jobs = [
        (seed, fold)
        for seed in ALL_SEEDS
        for fold in range(N_SPLITS)
    ]
    my_jobs = [job for index, job in enumerate(jobs) if index % WORLD_SIZE == RANK]

    rank_print(f"Assigned {len(my_jobs)} of {len(jobs)} seed/fold jobs: {my_jobs}")

    for seed, fold in my_jobs:
        output_path = job_prediction_path(seed, fold)
        if RESUME_COMPLETED_JOBS and output_path.is_file():
            try:
                existing = pd.read_csv(output_path, nrows=3)
                if set(PREDICTION_COLUMNS).issubset(existing.columns):
                    rank_print(f"Skipping completed seed={seed}, fold={fold}")
                    continue
            except Exception:
                pass

        set_seed(seed)
        rank_print(f"Starting seed={seed}, fold={fold}")

        validation_path = INTERMEDIATE_DIR / f"val_seed{seed}_fold{fold}.pkl"
        stage1_path = BASE_DIR_STAGE1 / f"seed{seed}_fold{fold}"
        stage2_path = BASE_DIR_STAGE2 / f"seed{seed}_fold{fold}"

        if not validation_path.is_file():
            raise FileNotFoundError(validation_path)

        validation_df = pd.read_pickle(validation_path)
        validation_df = prepare_validation_dataframe(validation_df, seed, fold)

        rank_print(f"Loading stage-1 adapter: {stage1_path}")
        stage1_model = load_adapter(stage1_path)
        validation_df = generate_stage1_predictions(
            validation_df, stage1_model, seed, fold
        )
        release_model(stage1_model)

        rank_print(f"Loading stage-2 adapter: {stage2_path}")
        stage2_model = load_adapter(stage2_path)
        fold_predictions = generate_stage2_predictions(
            validation_df, stage2_model, seed, fold
        )
        release_model(stage2_model)

        fold_auroc = roc_auc_score(
            fold_predictions["y_true"], fold_predictions["y_prob_raw"]
        )
        fold_brier = brier_score_loss(
            fold_predictions["y_true"], fold_predictions["y_prob_raw"]
        )

        temporary_path = output_path.with_suffix(".tmp")
        fold_predictions.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)

        rank_print(
            f"Completed seed={seed}, fold={fold}; "
            f"AUROC={fold_auroc:.4f}, Brier={fold_brier:.4f}; "
            f"saved {output_path}"
        )

        del fold_predictions, validation_df
        gc.collect()

def clip_probabilities(probabilities, epsilon=1e-6):
    return np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)

def fit_unpenalized_logistic(x, y):
    model = LogisticRegression(
        C=1e10,
        solver="lbfgs",
        max_iter=10000,
    )
    model.fit(np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y, dtype=int))
    return model

def calculate_calibration_statistics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = clip_probabilities(y_prob)
    predicted_logit = logit(y_prob)

    slope_model = fit_unpenalized_logistic(predicted_logit, y_true)
    calibration_intercept = float(slope_model.intercept_[0])
    calibration_slope = float(slope_model.coef_[0, 0])

    observed_prevalence = float(y_true.mean())

    def prevalence_difference(intercept):
        return float(expit(intercept + predicted_logit).mean() - observed_prevalence)

    citl = float(brentq(prevalence_difference, -100.0, 100.0))

    return {
        "CITL": citl,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "brier_score": brier_score_loss(y_true, y_prob),
    }

def fit_platt_scaler(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    predictors = logit(clip_probabilities(y_prob))
    return fit_unpenalized_logistic(predictors, y_true)

def apply_platt_scaler(calibrator, y_prob):
    predictors = logit(clip_probabilities(y_prob)).reshape(-1, 1)
    return calibrator.predict_proba(predictors)[:, 1]

def cross_fitted_platt_scaling(
    y_true,
    y_prob,
    groups=None,
    n_splits=5,
    random_state=2026,
):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = clip_probabilities(y_prob)
    calibrated = np.full(len(y_true), np.nan, dtype=float)

    if groups is not None:
        groups = np.asarray(groups)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        split_iterator = splitter.split(y_prob, y_true, groups=groups)
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        split_iterator = splitter.split(y_prob, y_true)

    for calibration_fold, (train_indices, test_indices) in enumerate(split_iterator):
        calibrator = fit_platt_scaler(
            y_true[train_indices], y_prob[train_indices]
        )
        calibrated[test_indices] = apply_platt_scaler(
            calibrator, y_prob[test_indices]
        )
        print(
            f"Platt fold {calibration_fold + 1}/{n_splits}: "
            f"train={len(train_indices)}, test={len(test_indices)}",
            flush=True,
        )

    if np.isnan(calibrated).any():
        raise RuntimeError("Some rows did not receive a cross-fitted probability.")
    return calibrated

def merge_prediction_files():
    expected_paths = [
        job_prediction_path(seed, fold)
        for seed in ALL_SEEDS
        for fold in range(N_SPLITS)
    ]
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The following seed/fold prediction files are missing after all "
            "ranks completed:\n" + "\n".join(missing)
        )

    frames = [pd.read_csv(path) for path in expected_paths]
    oof_df = pd.concat(frames, ignore_index=True)
    oof_df["patient_id"] = oof_df["patient_id"].astype(str)
    return oof_df

def validate_oof_data(oof_df):
    duplicate_keys = oof_df.duplicated(["patient_id", "seed"], keep=False)
    if duplicate_keys.any():
        duplicates = oof_df.loc[
            duplicate_keys, ["patient_id", "seed", "fold"]
        ].head(20)
        print(
            "WARNING: duplicate patient_id+seed rows detected. This may be valid "
            "for repeated encounters, but verify PATIENT_ID_COL:\n",
            duplicates.to_string(index=False),
            flush=True,
        )

    inconsistent = oof_df.groupby("patient_id")["y_true"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if len(inconsistent) > 0:
        raise ValueError(
            "Some patient IDs have inconsistent outcome labels across runs. "
            "Set PATIENT_ID_COL to the correct stable identifier."
        )

    unique_groups_per_class = (
        oof_df[["patient_id", "y_true"]]
        .drop_duplicates()
        .groupby("y_true")["patient_id"]
        .nunique()
    )
    if len(unique_groups_per_class) < 2:
        raise ValueError("Both outcome classes are required for calibration.")
    if int(unique_groups_per_class.min()) < PLATT_CV_SPLITS:
        raise ValueError(
            f"PLATT_CV_SPLITS={PLATT_CV_SPLITS}, but the smaller class has only "
            f"{int(unique_groups_per_class.min())} unique patient groups."
        )

def make_calibration_plot(patient_level_df, raw_stats, platt_stats, output_path):

    raw_color = "tab:orange"
    platt_color = "tab:green"
    perfect_color = "tab:blue"

    raw_fraction_positive, raw_mean_predicted = calibration_curve(
        y_true=patient_level_df["y_true"],
        y_prob=patient_level_df["y_prob_raw"],
        n_bins=N_CALIBRATION_BINS,
        strategy=CALIBRATION_BIN_STRATEGY,
    )
    platt_fraction_positive, platt_mean_predicted = calibration_curve(
        y_true=patient_level_df["y_true"],
        y_prob=patient_level_df["y_prob_platt"],
        n_bins=N_CALIBRATION_BINS,
        strategy=CALIBRATION_BIN_STRATEGY,
    )

    figure = plt.figure(figsize=(12.0, 5.6))
    calibration_axis = figure.add_axes([0.055, 0.105, 0.382667, 0.82])
    histogram_axis = figure.add_axes([0.530, 0.105, 0.382667, 0.82])

    calibration_axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=perfect_color,
        linewidth=1.5,
        label="Perfect calibration",
    )
    calibration_axis.plot(
        raw_mean_predicted,
        raw_fraction_positive,
        marker="o",
        color=raw_color,
        linewidth=2,
        label=(
            "Before calibration\n"
            f"CITL={raw_stats['CITL']:.4f}; "
            f"slope={raw_stats['calibration_slope']:.4f}; "
            f"Brier={raw_stats['brier_score']:.4f}"
        ),
    )
    calibration_axis.plot(
        platt_mean_predicted,
        platt_fraction_positive,
        marker="s",
        color=platt_color,
        linewidth=2,
        label=(
            "After Platt scaling\n"
            f"CITL={platt_stats['CITL']:.4f}; "
            f"slope={platt_stats['calibration_slope']:.4f}; "
            f"Brier={platt_stats['brier_score']:.4f}"
        ),
    )
    calibration_axis.set_xlabel("Mean predicted probability")
    calibration_axis.set_ylabel("Observed frequency")
    calibration_axis.set_title("Reliability diagram")
    calibration_axis.set_xlim(0, 1)
    calibration_axis.set_ylim(0, 1)
    calibration_axis.set_aspect("equal", adjustable="box")
    calibration_axis.set_box_aspect(1)
    calibration_axis.grid(alpha=0.3)
    calibration_axis.legend(loc="best", fontsize=9)

    histogram_bins = np.linspace(0, 1, 21)
    histogram_axis.hist(
        patient_level_df["y_prob_raw"],
        bins=histogram_bins,
        color=raw_color,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
        label="Before calibration",
    )
    histogram_axis.hist(
        patient_level_df["y_prob_platt"],
        bins=histogram_bins,
        color=platt_color,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
        label="After Platt scaling",
    )
    histogram_axis.set_xlabel("Predicted probability")
    histogram_axis.set_ylabel("Number of patients")
    histogram_axis.set_title("Predicted probability distribution")
    histogram_axis.set_xlim(0, 1)
    histogram_axis.set_box_aspect(1)
    histogram_axis.grid(alpha=0.3)
    histogram_axis.legend()

    png_path = output_path.with_suffix(".png")

    figure.savefig(
        png_path,
        dpi=300,
        format="png",
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(figure)

def rank_zero_analysis():
    print("\nAll ranks completed inference. Merging predictions...", flush=True)
    oof_df = merge_prediction_files()
    validate_oof_data(oof_df)

    raw_predictions_path = RESULTS_DIR / "stage2_oof_predictions_raw.csv"
    oof_df.to_csv(raw_predictions_path, index=False)

    print(f"OOF rows: {len(oof_df)}", flush=True)
    print(f"Unique patient IDs: {oof_df['patient_id'].nunique()}", flush=True)

    patient_level_df = (
        oof_df.groupby("patient_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_prob_raw=("y_prob_raw", "mean"),
            prediction_count=("y_prob_raw", "size"),
        )
    )

    patient_level_df["y_prob_platt"] = cross_fitted_platt_scaling(
        y_true=patient_level_df["y_true"].to_numpy(),
        y_prob=patient_level_df["y_prob_raw"].to_numpy(),
        groups=None,
        n_splits=PLATT_CV_SPLITS,
        random_state=2026,
    )
    patient_level_df["y_pred_platt"] = (
        patient_level_df["y_prob_platt"] >= 0.5
    ).astype(int)

    patient_probability_map = patient_level_df.set_index("patient_id")[
        "y_prob_platt"
    ]
    oof_df["y_prob_platt"] = oof_df["patient_id"].map(patient_probability_map)
    oof_df["y_pred_platt"] = (oof_df["y_prob_platt"] >= 0.5).astype(int)

    raw_stats = calculate_calibration_statistics(
        patient_level_df["y_true"], patient_level_df["y_prob_raw"]
    )
    platt_stats = calculate_calibration_statistics(
        patient_level_df["y_true"], patient_level_df["y_prob_platt"]
    )

    calibration_stats_df = pd.DataFrame(
        [
            {"model": "Before calibration", **raw_stats},
            {"model": "After Platt scaling", **platt_stats},
        ]
    )

    discrimination_df = pd.DataFrame(
        [
            {
                "model": "Before calibration",
                "AUROC": roc_auc_score(
                    patient_level_df["y_true"], patient_level_df["y_prob_raw"]
                ),
                "AUPRC": average_precision_score(
                    patient_level_df["y_true"], patient_level_df["y_prob_raw"]
                ),
            },
            {
                "model": "After Platt scaling",
                "AUROC": roc_auc_score(
                    patient_level_df["y_true"], patient_level_df["y_prob_platt"]
                ),
                "AUPRC": average_precision_score(
                    patient_level_df["y_true"], patient_level_df["y_prob_platt"]
                ),
            },
        ]
    )

    figure_path = RESULTS_DIR / "stage2_reliability_before_after_platt.png"
    make_calibration_plot(
        patient_level_df, raw_stats, platt_stats, figure_path
    )

    oof_output_path = RESULTS_DIR / "stage2_oof_predictions_with_platt.csv"
    patient_output_path = (
        RESULTS_DIR / "stage2_patient_level_predictions_with_platt.csv"
    )
    statistics_output_path = RESULTS_DIR / "stage2_calibration_statistics.csv"
    discrimination_output_path = (
        RESULTS_DIR / "stage2_discrimination_statistics.csv"
    )

    oof_df.to_csv(oof_output_path, index=False)
    patient_level_df.to_csv(patient_output_path, index=False)
    calibration_stats_df.to_csv(statistics_output_path, index=False)
    discrimination_df.to_csv(discrimination_output_path, index=False)

    final_platt_scaler = fit_platt_scaler(
        patient_level_df["y_true"].to_numpy(),
        patient_level_df["y_prob_raw"].to_numpy(),
    )
    final_platt_path = RESULTS_DIR / FINAL_SCALER_FILENAME
    joblib.dump(final_platt_scaler, final_platt_path)

    platt_intercept = float(final_platt_scaler.intercept_[0])
    platt_coefficient = float(final_platt_scaler.coef_[0, 0])
    platt_parameters_path = RESULTS_DIR / "stage2_final_platt_parameters.csv"
    pd.DataFrame(
        {
            "parameter": ["intercept", "coefficient"],
            "value": [platt_intercept, platt_coefficient],
        }
    ).to_csv(platt_parameters_path, index=False)

    run_metadata = {
        "world_size": WORLD_SIZE,
        "seeds": ALL_SEEDS,
        "n_model_folds": N_SPLITS,
        "platt_cv_splits": PLATT_CV_SPLITS,
        "calibration_bins": N_CALIBRATION_BINS,
        "calibration_bin_strategy": CALIBRATION_BIN_STRATEGY,
        "patient_id_column": PATIENT_ID_COL,
        "platt_intercept": platt_intercept,
        "platt_coefficient": platt_coefficient,
    }
    with open(RESULTS_DIR / "stage2_posthoc_run_metadata.json", "w") as file:
        json.dump(run_metadata, file, indent=2)

    print("\nCalibration statistics", flush=True)
    print(calibration_stats_df.to_string(index=False), flush=True)
    print("\nDiscrimination statistics", flush=True)
    print(discrimination_df.to_string(index=False), flush=True)
    print("\nFinal Platt equation:", flush=True)
    print(
        "calibrated_probability = sigmoid("
        f"{platt_intercept:.8f} + "
        f"{platt_coefficient:.8f} * logit(raw_probability))",
        flush=True,
    )
    print("\nSaved outputs:", flush=True)
    for path in [
        raw_predictions_path,
        oof_output_path,
        patient_output_path,
        statistics_output_path,
        discrimination_output_path,
        figure_path,
        final_platt_path,
        platt_parameters_path,
    ]:
        print(path, flush=True)

def main():
    try:
        run_inference_jobs()

        dist.barrier()

        dist.destroy_process_group()

        if IS_MAIN:
            rank_zero_analysis()
            rank_print("Post-hoc calibration completed successfully.")
        else:
            rank_print(
                "Inference outputs are complete; rank 0 is merging and calibrating."
            )

    finally:

        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
