# -*- coding: utf-8 -*-
"""
S_Acc Evaluation Framework for Chinese Idioms

Features:
- Content word F-beta (β=2) to reduce over-generation penalty
- Multi-reference aggregation: REF_AGG_MODE = "max" or "mean"
- BERTScore: enable/disable
- Accuracy metric: Cross-Encoder + Representation Ensemble + Fβ (three-layer hybrid, dominant layer has highest weight)

License: MIT
"""

import os
import re
import time
from typing import List, Tuple, Dict, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import jieba
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder

# ===== Configuration =====
# Reference explanation file (indicates language), actual model/output provided by environment variables
ORIGINAL_FILE = os.environ.get(
    "ORIGINAL_FILE",
    r"data/reference_explanations/chinese_explanations.csv"
)

# Input and output files controlled by batch_baseline.py
MODEL_FILE = os.environ.get("MODEL_FILE", "")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "results/sacc_chinese_output.csv")

# ===== Device Configuration =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== Model Configuration =====
# Use lightweight Chinese-specific model for acceleration (bge-small faster than large by 5-8x)
EMBED_MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
CROSS_ENCODER_MODEL = 'BAAI/bge-reranker-base'

BATCH_SIZE_EMB = 64
USE_BERTSCORE = True
BERTSCORE_LANG = 'zh'
MAX_LEN = 512

# ===== Aggregation and Fusion Parameters =====
BETA_F = 2.0
# REF_AGG_MODE controlled by environment variable (batch_baseline.py sets this)
REF_AGG_MODE = os.environ.get("REF_AGG_MODE", "max")  # "max" or "mean"
W_CROSS = 0.5
W_REP = 0.3
W_F = 0.2

# ===== Polarity Conflict Parameters =====
POLARITY_PENALTY = 0.5
POLARITY_MODE = "all"
POLARITY_RATIO = 0.5

# ===== Stopwords/Polarity Words (Chinese) =====
STOPWORDS: Set[str] = set("的了呢啊吧哦恩嗯亦也又与且在及或并被把对给而其之于") | \
    {"或者", "以及", "还有", "一种", "对于", "进行", "由于", "通过", "因此", "例如", "比如", "同时", "这", "是", "个", "来", "去", "为"}

NEGATIONS: Set[str] = {"不", "未", "无", "非", "别", "从不", "毫不", "不再", "不愿", "没", "没有", "不可", "难以", "未曾", "从未", "并非", "不是", "无法", "禁止", "否定", "反"}

POLAR_POS: Set[str] = {"喜爱", "喜欢", "爱慕", "钟爱", "偏爱", "珍爱", "喜好", "爱不释手", "舍不得", "不舍",
                       "珍视", "赞美", "欣赏", "褒扬", "优秀", "出色", "崇高", "优点", "优势", "肯定", "支持"}

POLAR_NEG: Set[str] = {"厌恶", "轻视", "不喜欢", "不珍惜", "舍弃", "抛弃", "丢弃", "贬低", "批评", "僵化", "死板", "落后",
                       "顽固", "不思进取", "空洞", "不正派", "失败", "倒退", "衰落"}

IDIOM_POLARITY_OVERRIDES: Dict[str, Dict[str, bool]] = {
    "爱不释手": {"pos": True,  "neg": False},
    "循规蹈矩": {"pos": False, "neg": True},
}

# ===== Basic Utilities =====
def clean_text(text: str) -> str:
    """Clean and normalize text input."""
    if pd.isna(text):
        return ""
    s = re.sub(r'[\[\]]', '', str(text))
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def tokenize_content(text: str) -> List[str]:
    """Tokenize text and remove stopwords."""
    return [t for t in jieba.cut(text) if t and t not in STOPWORDS and not t.isspace()]

REF_SPLIT_PAT = re.compile(r'\s*(\|\||;|；|/|、)\s*')
def to_refs(raw_cell: str) -> List[str]:
    """Parse reference explanations from raw cell content."""
    if pd.isna(raw_cell) or str(raw_cell).strip() == "":
        return []
    s = clean_text(str(raw_cell))
    parts = [p.strip() for p in re.split(REF_SPLIT_PAT, s)
             if p and p.strip() not in {"||", ";", "；", "/", "、"}]
    return parts if parts else [s]

# ===== F-beta Score =====
def fbeta_from_sets(pred_set: set, gold_set: set, beta: float = BETA_F) -> float:
    """
    Calculate F-beta score from token sets.

    Args:
        pred_set: Predicted token set
        gold_set: Gold reference token set
        beta: Beta parameter for F-beta score

    Returns:
        F-beta score
    """
    if not pred_set and not gold_set:
        return 1.0
    inter = len(pred_set & gold_set)
    if inter == 0:
        return 0.0
    prec = inter / max(1, len(pred_set))
    rec = inter / max(1, len(gold_set))
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0

def extract_polarity(raw_text: str, idiom: str = "") -> Tuple[bool, bool, bool]:
    """
    Extract polarity information from text.

    Args:
        raw_text: Text to analyze
        idiom: Idiom name for special overrides

    Returns:
        Tuple of (has_positive, has_negative, has_negation)
    """
    if idiom and idiom in IDIOM_POLARITY_OVERRIDES:
        ov = IDIOM_POLARITY_OVERRIDES[idiom]
        return bool(ov.get("pos", False)), bool(ov.get("neg", False)), False
    toks = list(jieba.cut(raw_text))
    has_pos = any(t in POLAR_POS for t in toks)
    has_neg = any(t in POLAR_NEG for t in toks)
    has_not = any(t in NEGATIONS for t in toks)
    return has_pos, has_neg, has_not

def _aggregate_list(vals: List[float], mode: str) -> float:
    """Aggregate list of values using mean or max."""
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals)) if mode == "mean" else float(np.max(vals))

def content_fbeta_with_polarity_multi(pred_text: str, gold_refs: List[str], idiom: str,
                                     beta: float = BETA_F) -> Tuple[float, bool]:
    """
    Calculate content F-beta with polarity conflict detection.

    Args:
        pred_text: Predicted text
        gold_refs: List of gold reference texts
        idiom: Idiom name
        beta: Beta parameter

    Returns:
        Tuple of (aggregated_fbeta_score, has_polarity_conflict)
    """
    pred_set = set(tokenize_content(pred_text))
    if not gold_refs:
        return 0.0, False

    p_pos, p_neg, p_not = extract_polarity(pred_text, idiom)
    scores, conflicts = [], []

    for g_raw in gold_refs:
        g_set = set(tokenize_content(g_raw))
        fb = fbeta_from_sets(pred_set, g_set, beta)
        g_pos, g_neg, g_not = extract_polarity(g_raw, idiom)

        pol_conf = False
        if g_pos or g_neg or p_pos or p_neg:
            if g_pos and (p_neg or (p_not and not p_pos)):
                pol_conf = True
            elif g_neg and (p_pos and not p_not):
                pol_conf = True

        scores.append(fb)
        conflicts.append(pol_conf)

    fb_agg = _aggregate_list(scores, REF_AGG_MODE)

    if POLARITY_MODE == "all":
        pol_conf_final = all(conflicts)
    elif POLARITY_MODE == "any":
        pol_conf_final = any(conflicts)
    elif POLARITY_MODE == "ratio":
        pol_conf_final = (sum(conflicts) / max(1, len(conflicts))) >= POLARITY_RATIO
    else:
        pol_conf_final = all(conflicts)

    if pol_conf_final:
        fb_agg *= POLARITY_PENALTY

    return fb_agg, pol_conf_final

# ===== Cross-Encoder =====
def load_cross_encoder(name: str) -> CrossEncoder:
    """Load Cross-Encoder model for relevance scoring."""
    print(f"[CE] Loading {name} ...")
    try:
        model = CrossEncoder(name, device=DEVICE)
        print("[CE] Model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] CrossEncoder load failed: {e}")
        return None

def cross_encoder_over_refs(ce_model: CrossEncoder, pred: str, refs: List[str]) -> float:
    """Calculate Cross-Encoder scores across multiple references."""
    if ce_model is None or not refs:
        return np.nan
    pairs = [(clean_text(r), clean_text(pred)) for r in refs]
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs:
        return 0.0
    scores = ce_model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    probs = torch.sigmoid(torch.tensor(scores)).numpy()
    return float(np.mean(probs)) if REF_AGG_MODE == "mean" else float(np.max(probs))

# ===== Semantic Similarity =====
def load_embed_model(name: str) -> SentenceTransformer:
    """Load sentence embedding model."""
    print(f"[STS] Loading {name} ...")
    try:
        model = SentenceTransformer(name, device=DEVICE)
        print("[STS] Model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] STS model load failed: {e}")
        return None

def sts_over_refs(embed_model: SentenceTransformer, pred: str, refs: List[str],
                  batch_size: int = BATCH_SIZE_EMB) -> float:
    """Calculate semantic similarity using sentence embeddings."""
    if embed_model is None or not refs:
        return np.nan
    clean_refs = [clean_text(r) for r in refs if clean_text(r)]
    clean_pred = clean_text(pred)
    if not clean_refs or not clean_pred:
        return 0.0
    v_pred = embed_model.encode([clean_pred], batch_size=1, convert_to_numpy=True, show_progress_bar=False)
    v_refs = embed_model.encode(clean_refs, batch_size=min(len(clean_refs), batch_size),
                                convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity(v_pred, v_refs)[0]
    return float(np.mean(sims)) if REF_AGG_MODE == "mean" else float(np.max(sims))

def bertscore_over_refs(pred: str, refs: List[str], lang: str = 'zh',
                       batch_size: int = 32, rescale: bool = False) -> float:
    """Calculate BERTScore across multiple references."""
    if not refs:
        return np.nan
    try:
        from bert_score import score as bert_score_fn
        clean_refs = [clean_text(r) for r in refs if clean_text(r)]
        clean_pred = clean_text(pred)
        if not clean_refs or not clean_pred:
            return 0.0
        P, R, F1 = bert_score_fn(
            [clean_pred] * len(clean_refs),
            clean_refs,
            lang=lang,
            batch_size=batch_size,
            verbose=False,
            rescale_with_baseline=rescale,
            device=DEVICE,
        )
        F1_np = F1.numpy()
        return float(np.mean(F1_np)) if REF_AGG_MODE == "mean" else float(F1_np.max())
    except Exception as e:
        print(f"[WARN] BERTScore failed: {e}")
        return np.nan

# ===== Lexical Cosine Similarity =====
def lexical_cosine_over_refs(pred: str, refs: List[str]) -> float:
    """Calculate lexical cosine similarity."""
    tokens_pred = tokenize_content(pred)
    if not refs or not tokens_pred:
        return np.nan
    vec_pred = Counter(tokens_pred)

    def cos_with(ref_tokens):
        vec_ref = Counter(ref_tokens)
        inter = sum(vec_pred[t] * vec_ref[t] for t in vec_pred.keys() & vec_ref.keys())
        norm_pred = np.sqrt(sum(v * v for v in vec_pred.values()))
        norm_ref = np.sqrt(sum(v * v for v in vec_ref.values()))
        if norm_pred == 0 or norm_ref == 0:
            return 0.0
        return inter / (norm_pred * norm_ref)

    vals = []
    for r in refs:
        toks = tokenize_content(r)
        if toks:
            vals.append(cos_with(toks))
    if not vals:
        return 0.0
    return float(np.mean(vals)) if REF_AGG_MODE == "mean" else float(np.max(vals))

# ===== Hierarchical Fusion =====
def representation_ensemble(bert_val: float, sts_val: float, lex_val: float) -> float:
    """Ensemble different representation-based scores."""
    vals = [v for v in [bert_val, sts_val, lex_val] if not np.isnan(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))

def calculate_accuracy_score(ce_val: float, rep_val: float, fbeta_val: float) -> float:
    """Calculate final S_Acc score using weighted ensemble."""
    ce_val = 0.0 if np.isnan(ce_val) else ce_val
    rep_val = 0.0 if np.isnan(rep_val) else rep_val
    fbeta_val = 0.0 if np.isnan(fbeta_val) else fbeta_val
    return W_CROSS * ce_val + W_REP * rep_val + W_F * fbeta_val

# ===== Main Processing =====
def main():
    """Main evaluation function."""
    t0 = time.time()
    print(f"DEVICE = {DEVICE}")

    if not MODEL_FILE or not OUTPUT_FILE:
        print("[ERROR] MODEL_FILE and OUTPUT_FILE must be set via environment variables")
        return

    # Load data
    original_df = pd.read_csv(ORIGINAL_FILE, encoding='utf-8-sig')
    # For CSV format, we need to load multiple files corresponding to Excel sheets
    model_file_path = Path(MODEL_FILE)
    model_dir = model_file_path.parent
    model_base = model_file_path.stem

    # Find all CSV files that correspond to the Excel sheets
    model_sheets = {}
    for csv_file in model_dir.glob(f"{model_base}_*.csv"):
        sheet_name = csv_file.stem.replace(f"{model_base}_", "")
        model_sheets[sheet_name] = pd.read_csv(csv_file, encoding='utf-8-sig')

    # Load models
    embed_model = load_embed_model(EMBED_MODEL_NAME)
    cross_encoder = load_cross_encoder(CROSS_ENCODER_MODEL)

    # Process reference explanations
    idioms = original_df.iloc[:, 0].astype(str).tolist()
    ref_lists: List[List[str]] = []
    for i in range(len(original_df)):
        refs = to_refs(original_df.iloc[i, 1])
        if original_df.shape[1] > 2:
            refs += to_refs(original_df.iloc[i, 2])
        refs = [clean_text(r) for r in refs if r and clean_text(r)]
        if not refs:
            refs = [""]
        ref_lists.append(refs)

    # Process each sheet
    avg_rows = []
    # For CSV format, save each sheet as separate CSV file
    output_path = Path(OUTPUT_FILE)
    output_dir = output_path.parent
    output_base = output_path.stem

    for sheet_name, sheet_df in tqdm(model_sheets.items(), desc="Processing sheets"):
        if len(sheet_df) != len(original_df):
            print(f"[WARN] Sheet {sheet_name} has mismatched row count, skipping")
            continue

        preds = [clean_text(t) for t in sheet_df.iloc[:, 1].tolist()]
        rows = []

        for idiom, refs, pred in zip(idioms, ref_lists, preds):
            # Calculate various scores
            ce_val = cross_encoder_over_refs(cross_encoder, pred, refs)
            sts_val = sts_over_refs(embed_model, pred, refs)
            bert_val = bertscore_over_refs(pred, refs, lang=BERTSCORE_LANG) if USE_BERTSCORE else np.nan
            lex_val = lexical_cosine_over_refs(pred, refs)
            rep_val = representation_ensemble(bert_val, sts_val, lex_val)
            fbeta_val, pol_conf = content_fbeta_with_polarity_multi(pred, refs, idiom=idiom)
            score_acc = calculate_accuracy_score(ce_val, rep_val, fbeta_val)

            rows.append({
                'Idiom': idiom,
                'Reference': " || ".join(refs),
                f'{sheet_name}_Prediction': pred,
                f'{sheet_name}_CrossEncoder': ce_val,
                f'{sheet_name}_BERT_F1': bert_val,
                f'{sheet_name}_STS': sts_val,
                f'{sheet_name}_LexicalCos': lex_val,
                f'{sheet_name}_Representation': rep_val,
                f'{sheet_name}_F1_Content': fbeta_val,
                f'{sheet_name}_PolarityConflict': pol_conf,
                f'{sheet_name}_S_Acc': score_acc,
            })

        df = pd.DataFrame(rows)
        # Save individual sheet data as CSV
        sheet_csv = output_dir / f"{output_base}_{sheet_name}.csv"
        df.to_csv(sheet_csv, index=False, encoding='utf-8-sig')

        # Calculate averages
        avg_rows.append({
            'Arrangement': sheet_name,
            'Avg_CrossEncoder': df[f'{sheet_name}_CrossEncoder'].replace([np.inf, -np.inf], np.nan).mean(),
            'Avg_BERT_F1': df[f'{sheet_name}_BERT_F1'].replace([np.inf, -np.inf], np.nan).mean(),
            'Avg_STS': df[f'{sheet_name}_STS'].replace([np.inf, -np.inf], np.nan).mean(),
            'Avg_LexicalCos': df[f'{sheet_name}_LexicalCos'].replace([np.inf, -np.inf], np.nan).mean(),
            'Avg_Representation': df[f'{sheet_name}_Representation'].replace([np.inf, -np.inf], np.nan).mean(),
            'Avg_F1_Content': df[f'{sheet_name}_F1_Content'].replace([np.inf, -np.inf], np.nan).mean(),
            'Avg_S_Acc': df[f'{sheet_name}_S_Acc'].replace([np.inf, -np.inf], np.nan).mean(),
        })

    # Write summary CSV
    summary_csv = output_dir / f"{output_base}_Summary.csv"
    pd.DataFrame(avg_rows).to_csv(summary_csv, index=False, encoding='utf-8-sig')

    print(f"[DONE] Output: {OUTPUT_FILE}, Time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
