#!/usr/bin/env python3
"""
validar_dataset.py — Popula Milvus e avalia modelos no dataset_ext_val_lfw
===========================================================================
Fluxo automático:
  1. Descobre quais modelos têm arquivo de pesos disponível
  2. Para cada modelo, verifica se a collection _lfw_ext_val está populada
     com a quantidade certa de imagens — se não, popula agora
  3. Avalia cada modelo com leave-one-out nos dois cenários:
       Caso 1 — identidades completas (>= --complete imgs)
       Caso 2 — todas as identidades
  4. Gera avaliacao_modelos.md com tabela comparativa

Uso:
  python validar_dataset.py
  python validar_dataset.py --dataset-dir dataset_ext_val_lfw
  python validar_dataset.py --models topofr_r100_glint lvface_b_glint
  python validar_dataset.py --recreate          # força re-população
  python validar_dataset.py --skip-populate     # pula fase 1
  python validar_dataset.py --complete 3        # threshold "completo"
"""

import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Silenciar FutureWarning de modulos third-party (deprecations upstream que
# nao afetam a execucao). Mantemos warnings de outros pacotes visiveis.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"face_module\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.config import Config
from models import ModelFactory

try:
    from preprocessing import extract_embedding_standardized
    from utils.face_detection import NoFaceDetectedError
except ImportError:
    NoFaceDetectedError = Exception

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

import torch

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset_ext_val_lfw"
DEFAULT_REPORT_PATH = SCRIPT_DIR / "avaliacao_modelos.md"
RESULTS_DIR         = SCRIPT_DIR / "resultados"
VALID_EXT           = {".jpg", ".jpeg", ".png"}
DEFAULT_COMPLETE    = 5
DEFAULT_MIN_EVAL    = 2   # min imgs/id para entrar no cenario "todas"
TOP_K               = 10
BATCH_SIZE          = 64
COLL_SUFFIX         = "_lfw_ext_val"   # distingue do _ext_val do dataset antigo

logging.basicConfig(level=logging.CRITICAL)
log = logging.getLogger("validar")
log.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_h)
log.propagate = False


# ===========================================================================
# Utilitários
# ===========================================================================

def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def scan_dataset(dataset_dir: Path) -> Dict[str, List[Path]]:
    skip = {"_tmp", "_tmp_crawl", "_tmp_recrawl", "_tmp_wm", "__pycache__"}
    result = {}
    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir() or d.name in skip or d.name.startswith("."):
            continue
        imgs = sorted(f for f in d.iterdir()
                      if f.suffix.lower() in VALID_EXT and f.is_file())
        if imgs:
            result[d.name] = imgs
    return result


def models_with_weights() -> List[str]:
    """Retorna modelos cujo arquivo de pesos existe em disco."""
    available = []
    for name, path in Config.MODEL_WEIGHTS.items():
        p = Path(path)
        if p.exists():
            available.append(name)
        else:
            log.info(f"  {name}: peso não encontrado ({p.name})")
    return available


def collection_name(model_key: str) -> str:
    return Config.get_collection_name(model_key) + COLL_SUFFIX


def _raw_client():
    from pymilvus import MilvusClient as PyMilvus
    return PyMilvus(Config.MILVUS_DB_PATH)


def _row_count(raw, coll: str) -> int:
    if not raw.has_collection(coll):
        return -1
    try:
        return int(raw.get_collection_stats(coll).get("row_count", 0))
    except Exception:
        return 0


def _manual_auc(labels, scores):
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    tp_total = sum(labels)
    fp_total = len(labels) - tp_total
    if tp_total == 0 or fp_total == 0:
        return 0.0
    auc = prev_fpr = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp / tp_total) * (fp / fp_total - prev_fpr)
            prev_fpr = fp / fp_total
    return auc


# ===========================================================================
# Fase 1 — Popular Milvus
# ===========================================================================

def populate(
    model_key: str,
    identities: Dict[str, List[Path]],
    coll: str,
    recreate: bool,
) -> int:
    """Popula a collection com embeddings do dataset. Retorna nº de inserções."""
    from app.milvus_client import MilvusClient

    Config.USE_TTA = True
    Config.EMBEDDING_DIM = 1024

    model = ModelFactory.create(model_key, use_tta=True, use_cache=False)
    milvus = MilvusClient(collection_name=coll)

    if recreate:
        milvus.recreate_collection()

    batch_e, batch_p, batch_i = [], [], []
    inserted = errors = 0

    all_images = [(ident, p) for ident, imgs in identities.items() for p in imgs]

    for identity, img_path in tqdm(all_images, desc=f"  Populando {model_key}", unit="img"):
        try:
            emb = extract_embedding_standardized(
                model, file_path=str(img_path), use_tta=True,
                use_face_detection=Config.USE_FACE_DETECTION,
                conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD,
                select_largest=Config.FACE_DETECTION_SELECT_LARGEST,
            )
            batch_e.append(emb)
            batch_p.append(identity)
            batch_i.append(str(img_path))

            if len(batch_e) >= BATCH_SIZE:
                milvus.insert(batch_e, batch_p, batch_i)
                inserted += len(batch_e)
                batch_e, batch_p, batch_i = [], [], []

        except (NoFaceDetectedError, Exception):
            errors += 1

    if batch_e:
        milvus.insert(batch_e, batch_p, batch_i)
        inserted += len(batch_e)

    log.info(f"  {model_key}: {inserted} inseridos, {errors} erros")

    # Liberar GPU
    ModelFactory.clear_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return inserted


# ===========================================================================
# Fase 2 — Avaliação leave-one-out
# ===========================================================================

def evaluate(
    model_key: str,
    identities: Dict[str, List[Path]],
    coll: str,
) -> Dict:
    """Leave-one-out: query cada imagem, exclui self por image_path, computa métricas."""
    from app.milvus_client import MilvusClient

    model = ModelFactory.create(model_key, use_tta=True, use_cache=False)
    milvus = MilvusClient(collection_name=coll)

    all_queries = [(ident, p) for ident, imgs in identities.items() for p in imgs]

    hits1 = hits5 = 0
    rr_list, p5_list = [], []
    genuine_sims, impostor_sims = [], []
    all_labels, all_scores = [], []
    errors = 0

    for identity, img_path in tqdm(all_queries, desc=f"  Avaliando {model_key}", unit="img"):
        try:
            emb = extract_embedding_standardized(
                model, file_path=str(img_path), use_tta=True,
                use_face_detection=Config.USE_FACE_DETECTION,
                conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD,
                select_largest=Config.FACE_DETECTION_SELECT_LARGEST,
            )
        except (NoFaceDetectedError, Exception):
            errors += 1
            continue

        results = milvus.search(query_embedding=emb, top_k=TOP_K,
                                output_fields=["person_id", "image_path"])

        # Excluir self-match pelo caminho da imagem
        filtered = [r for r in results if r.get("image_path", "") != str(img_path)]
        if not filtered:
            errors += 1
            continue

        top5 = filtered[:5]

        if filtered[0]["person_id"] == identity:
            hits1 += 1
        if any(r["person_id"] == identity for r in top5):
            hits5 += 1

        rr = 0.0
        for idx, r in enumerate(filtered):
            if r["person_id"] == identity:
                rr = 1.0 / (idx + 1)
                break
        rr_list.append(rr)

        correct_in_5 = sum(1 for r in top5 if r["person_id"] == identity)
        p5_list.append(correct_in_5 / 5.0)

        for r in filtered[:TOP_K - 1]:
            sim = float(r.get("distance", 0.0))
            is_gen = r["person_id"] == identity
            all_scores.append(sim)
            all_labels.append(1 if is_gen else 0)
            (genuine_sims if is_gen else impostor_sims).append(sim)

    ModelFactory.clear_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(all_queries) - errors
    hr1 = hits1 / max(n, 1)
    hr5 = hits5 / max(n, 1)
    mrr = float(np.mean(rr_list)) if rr_list else 0.0
    p5  = float(np.mean(p5_list)) if p5_list else 0.0
    sim_gen = float(np.mean(genuine_sims)) if genuine_sims else 0.0
    sim_imp = float(np.mean(impostor_sims)) if impostor_sims else 0.0
    gap = sim_imp - sim_gen

    auc = 0.0
    if all_labels and len(set(all_labels)) > 1:
        if SKLEARN_OK:
            try:
                auc = roc_auc_score(all_labels, all_scores)
            except Exception:
                pass
        else:
            auc = _manual_auc(all_labels, all_scores)

    return {
        "model":   model_key,
        "n":       n,
        "errors":  errors,
        "HR1":     hr1,
        "HR5":     hr5,
        "MRR":     mrr,
        "P5":      p5,
        "AUC":     auc,
        "sim_gen": sim_gen,
        "sim_imp": sim_imp,
        "gap":     gap,
    }


# ===========================================================================
# Fase 3 — Relatório Markdown
# ===========================================================================

def _table_row(m: str, r: Dict, top_k: int) -> str:
    return (
        f"| `{m}` "
        f"| **{_pct(r['HR1'])}** "
        f"| {_pct(r['HR5'])} "
        f"| {r['MRR']:.4f} "
        f"| {r['AUC']:.4f} "
        f"| {r['sim_gen']:.4f} "
        f"| {r['gap']:+.4f} "
        f"| {r['n']:,} |"
    )


def generate_report(
    results_complete: Dict[str, Dict],
    results_all: Dict[str, Dict],
    meta: Dict,
    complete_threshold: int,
    min_eval_threshold: int,
    report_path: Path,
):
    order = sorted(results_complete, key=lambda m: results_complete[m]["HR1"], reverse=True)

    header  = f"| Modelo | HR@1 | HR@5 | MRR | AUC | Sim Gen | Gap | Queries |"
    divider = f"|--------|------|------|-----|-----|---------|-----|---------|"

    lines = [
        f"# Avaliação de Modelos — Dataset de Validação Externa",
        f"",
        f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Dataset:** `dataset_ext_val_lfw/`  ",
        f"**Avaliação:** leave-one-out nas collections `{COLL_SUFFIX}`  ",
        f"**Banco Milvus:** `{Path(Config.MILVUS_DB_PATH).name}`",
        f"",
        f"---",
        f"",
        f"## Caso 1 — Identidades Completas (≥ {complete_threshold} imagens)",
        f"",
        f"_{meta['n_ids_complete']:,} identidades · {meta['n_imgs_complete']:,} imagens_",
        f"",
        header, divider,
    ]
    for m in order:
        if m in results_complete:
            lines.append(_table_row(m, results_complete[m], TOP_K))

    lines += [
        f"",
        f"---",
        f"",
        f"## Caso 2 — Identidades com ≥ {min_eval_threshold} imagens",
        f"",
        f"_{meta['n_ids_all']:,} identidades · {meta['n_imgs_all']:,} imagens_  ",
        f"_(excluídas {meta['n_excluded']:,} identidades com < {min_eval_threshold} imagens)_",
        f"",
        header, divider,
    ]
    for m in order:
        if m in results_all:
            lines.append(_table_row(m, results_all[m], TOP_K))

    lines += [
        f"",
        f"---",
        f"",
        f"## Legenda",
        f"",
        f"| Métrica | Descrição |",
        f"|---------|-----------|",
        f"| **HR@1** | Fração de queries onde o top-1 é da identidade correta |",
        f"| **HR@5** | Idem para os 5 primeiros resultados |",
        f"| **MRR** | Mean Reciprocal Rank — quão cedo a resposta correta aparece |",
        f"| **AUC** | Area Under ROC Curve (pares genuínos vs impostores) |",
        f"| **Sim Gen** | Similaridade coseno média entre pares genuínos |",
        f"| **Gap** | Sim impostor − Sim genuíno — negativo é bom |",
        f"| **Queries** | Imagens avaliadas (excluindo erros de extração) |",
        f"",
        f"> Avaliação **leave-one-out**: cada imagem é usada como query, o "
        f"self-match é removido pelo `image_path`, e o rank-1 restante é verificado.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Popula Milvus e avalia modelos no dataset_ext_val_lfw"
    )
    p.add_argument("--dataset-dir",  type=str, default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--report",       type=str, default=str(DEFAULT_REPORT_PATH))
    p.add_argument("--models",       nargs="+", default=None,
                   help="Modelos a avaliar (default: todos com peso disponível)")
    p.add_argument("--complete",     type=int, default=DEFAULT_COMPLETE,
                   help=f"Threshold de imagens para identidade 'completa' (default: {DEFAULT_COMPLETE})")
    p.add_argument("--min-images-eval", type=int, default=DEFAULT_MIN_EVAL,
                   help=(
                       f"Min imgs/id no cenario 'todas as identidades' "
                       f"(default: {DEFAULT_MIN_EVAL}). Identidades com menos "
                       f"que isso sao excluidas — leave-one-out exige >=2."
                   ))
    p.add_argument("--recreate",     action="store_true",
                   help="Recriar collections (força re-população)")
    p.add_argument("--skip-populate",action="store_true",
                   help="Pular fase 1 — usar collections existentes")
    p.add_argument("-v", "--verbose",action="store_true")
    args = p.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    dataset_dir = Path(args.dataset_dir)
    report_path = Path(args.report)

    if not dataset_dir.exists():
        print(f"ERRO: dataset não encontrado: {dataset_dir}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Escanear dataset
    # -----------------------------------------------------------------------
    print("=" * 68)
    print("  AVALIAÇÃO — Dataset de Validação Externa (LFW)")
    print("=" * 68)

    log.info("Escaneando dataset...")
    all_ids_raw  = scan_dataset(dataset_dir)
    all_ids      = {k: v for k, v in all_ids_raw.items() if len(v) >= args.min_images_eval}
    complete_ids = {k: v for k, v in all_ids_raw.items() if len(v) >= args.complete}

    n_imgs_raw      = sum(len(v) for v in all_ids_raw.values())
    n_imgs_all      = sum(len(v) for v in all_ids.values())
    n_imgs_complete = sum(len(v) for v in complete_ids.values())
    n_excluded      = len(all_ids_raw) - len(all_ids)

    print(f"  Dataset:                         {dataset_dir.name}/")
    print(f"  Identidades brutas (>=1 img):    {len(all_ids_raw):,}  ({n_imgs_raw:,} imagens)")
    print(f"  Excluidas do eval (<{args.min_images_eval} imgs):     {n_excluded:,}")
    print(f"  Avaliaveis (>={args.min_images_eval} imgs):           {len(all_ids):,}  ({n_imgs_all:,} imagens)")
    print(f"  Completas (>={args.complete} imgs):              {len(complete_ids):,}  ({n_imgs_complete:,} imagens)")
    print()

    # -----------------------------------------------------------------------
    # Detectar modelos com pesos disponíveis
    # -----------------------------------------------------------------------
    candidates = args.models or models_with_weights()
    if not candidates:
        print("Nenhum modelo com peso encontrado.")
        sys.exit(1)

    raw = _raw_client()
    models_to_run: List[Tuple[str, str, bool]] = []  # (model_key, coll, needs_populate)

    print(f"  {'Modelo':<30s} {'Collection':<45s} {'Rows':>8s}  {'Ação'}")
    print(f"  {'-'*30} {'-'*45} {'-'*8}  {'-'*12}")
    for mk in candidates:
        coll = collection_name(mk)
        rows = _row_count(raw, coll)
        needs = rows < n_imgs_all * 0.95 or args.recreate   # tolera ±5% de erros
        action = "POPULAR" if needs and not args.skip_populate else "OK"
        print(f"  {mk:<30s} {coll:<45s} {max(rows,0):>8,}  {action}")
        models_to_run.append((mk, coll, needs))

    print()

    # -----------------------------------------------------------------------
    # Fase 1: Popular collections que precisam
    # -----------------------------------------------------------------------
    if not args.skip_populate:
        need_pop = [(mk, coll) for mk, coll, needs in models_to_run if needs]
        if need_pop:
            print("=" * 68)
            print("  FASE 1 — Populando Collections")
            print("=" * 68)
            for mk, coll in need_pop:
                print(f"\n  [{mk}]")
                populate(mk, all_ids, coll, recreate=args.recreate)
        else:
            print("  Fase 1: todas as collections já estão populadas.")
    else:
        print("  Fase 1: pulada (--skip-populate).")

    print()

    # -----------------------------------------------------------------------
    # Fase 2: Avaliação leave-one-out
    # -----------------------------------------------------------------------
    print("=" * 68)
    print("  FASE 2 — Avaliação Leave-One-Out")
    print("=" * 68)

    results_complete: Dict[str, Dict] = {}
    results_all:      Dict[str, Dict] = {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for mk, coll, _ in models_to_run:
        print(f"\n  [{mk}]")

        # Verificar se collection tem dados suficientes
        rows = _row_count(_raw_client(), coll)
        if rows <= 0:
            print(f"  Collection vazia — pulando.")
            continue

        print(f"  Cenário 1: identidades completas...")
        r1 = evaluate(mk, complete_ids, coll)
        results_complete[mk] = r1
        print(f"    HR@1={_pct(r1['HR1'])}  HR@5={_pct(r1['HR5'])}  "
              f"MRR={r1['MRR']:.4f}  AUC={r1['AUC']:.4f}  "
              f"n={r1['n']:,}  erros={r1['errors']}")

        print(f"  Cenário 2: identidades com >={args.min_images_eval} imagens...")
        r2 = evaluate(mk, all_ids, coll)
        results_all[mk] = r2
        print(f"    HR@1={_pct(r2['HR1'])}  HR@5={_pct(r2['HR5'])}  "
              f"MRR={r2['MRR']:.4f}  AUC={r2['AUC']:.4f}  "
              f"n={r2['n']:,}  erros={r2['errors']}")

        # Salvar métricas individuais
        date_str = datetime.now().strftime("%Y%m%d")
        json_path = RESULTS_DIR / f"metricas_{mk}_lfw_ext_val_{date_str}.json"
        json_path.write_text(
            json.dumps({"complete": r1, "all": r2, "ts": datetime.now().isoformat()},
                       indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # -----------------------------------------------------------------------
    # Fase 3: Relatório Markdown
    # -----------------------------------------------------------------------
    print()
    print("=" * 68)
    print("  FASE 3 — Relatório")
    print("=" * 68)

    meta = {
        "n_ids_all":       len(all_ids),
        "n_imgs_all":      n_imgs_all,
        "n_ids_complete":  len(complete_ids),
        "n_imgs_complete": n_imgs_complete,
        "n_excluded":      n_excluded,
    }

    generate_report(
        results_complete, results_all, meta,
        args.complete, args.min_images_eval, report_path,
    )
    print(f"\n  Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
