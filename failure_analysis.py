#!/usr/bin/env python3
"""
failure_analysis.py
====================
Identifica imagens com pior desempenho (HR@1) em uma collection Milvus.

Metodologia: Leave-one-out
  Para cada embedding no Milvus, busca top_k vizinhos, filtra o self-match
  por image_path, e verifica se o top-1 restante pertence à mesma identidade.

Saída:
  1. Relatório CSV com detalhes por imagem (failures + successes)
  2. Sumário por identidade (% HR@1 por pessoa)
  3. Confusion pairs (pares de identidades mais confundidos)
  4. Resumo no terminal
  5. (Opcional) Relatório visual HTML com imagens lado a lado

Uso:
  python failure_analysis.py --model topofr_r100_glint
  python failure_analysis.py --model topofr_r100_glint --top-n 50
  python failure_analysis.py --model topofr_r100_glint --output-dir ./analysis_results
  python failure_analysis.py --model topofr_r100_glint --only-failures
  python failure_analysis.py --model topofr_r100_glint --visual-report
  python failure_analysis.py --model topofr_r100_glint --visual-report --max-visual 100
"""

import os
import sys
import argparse
import json
import csv
import base64
import html as html_module
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import Config
from app.milvus_client import MilvusClient


# =========================================================================
# Configurações
# =========================================================================
MILVUS_QUERY_BATCH_SIZE = 300     # Milvus Lite limita ~64MB por query; 300×1024×4B≈1.2MB
SEARCH_TOP_K = 11                 # top_k para busca (10 + 1 self-match)
MAX_RANK_TRACK = 10               # Rastrear rank do match correto até rank N


def parse_args():
    parser = argparse.ArgumentParser(
        description="Análise de falhas HR@1 por imagem no Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python failure_analysis.py --model topofr_r100_glint
  python failure_analysis.py --model topofr_r100_glint --collection face_embeddings_topofr_r100_glint_ext_val
  python failure_analysis.py --model cosface_resnet50 --top-n 30
  python failure_analysis.py --model topofr_r100_glint --only-failures --output-dir ./reports
        """
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=Config.AVAILABLE_MODELS,
        help="Nome do modelo (determina a collection Milvus)"
    )
    parser.add_argument(
        "--collection", type=str, default=None,
        help="Nome da collection Milvus (sobrescreve o mapeamento automático do --model)"
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Mostrar apenas os N piores resultados no terminal (default: todos os failures)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Diretório para salvar relatórios (default: ./failure_analysis_<model>_<timestamp>)"
    )
    parser.add_argument(
        "--only-failures", action="store_true",
        help="Exportar no CSV apenas as imagens que falharam HR@1"
    )
    parser.add_argument(
        "--search-top-k", type=int, default=SEARCH_TOP_K,
        help=f"top_k para busca no Milvus (default: {SEARCH_TOP_K})"
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Caminho para o arquivo .db do Milvus (default: config padrão)"
    )
    parser.add_argument(
        "--visual-report", action="store_true",
        help="Gerar relatório visual HTML com imagens lado a lado"
    )
    parser.add_argument(
        "--max-visual", type=int, default=200,
        help="Máximo de falhas a incluir no relatório visual (default: 200)"
    )
    return parser.parse_args()


# =========================================================================
# Funções de consulta ao Milvus
# =========================================================================

def fetch_all_records(client: MilvusClient, collection_name: str) -> List[Dict]:
    """
    Milvus Lite tem dois limites que impedem paginacao por offset em colecoes grandes:
      1. Limite de tamanho de resposta (~64 MB): 16384 registros x 1024 floats
         x 4 B = 64 MB bate exatamente no teto interno do segcore.
      2. offset + limit <= 16384: impossibilita acessar registros alem da posicao
         16383 com qualquer batch size.

    A paginacao por cursor (filter="id > last_id") contorna ambos porque:
      - Batches pequenos (MILVUS_QUERY_BATCH_SIZE registros) ficam bem abaixo
        do limite de tamanho.
      - Nenhum offset e usado; o filtro de ID avanca o cursor a cada iteracao.
    """
    all_records = []

    # Obter total para progresso -----------------------------------------------
    total = 0
    try:
        raw_stats = client.client.get_collection_stats(collection_name)
        if isinstance(raw_stats, dict):
            total = int(raw_stats.get("row_count", 0))
    except Exception:
        pass

    # Fallback: wrapper do app usa chave "total_embeddings"
    if total == 0:
        try:
            app_stats = client.get_collection_stats()
            if isinstance(app_stats, dict):
                total = int(app_stats.get("total_embeddings", 0))
        except Exception:
            pass

    if total == 0:
        print(f"⚠ Collection '{collection_name}' esta vazia ou inacessivel!")
        try:
            available = client.client.list_collections()
            print(f"  Collections disponiveis: {available}")
            print(f"  Dica: use --collection <nome> para especificar outra collection.")
        except Exception:
            pass
        return []

    print(f"Buscando {total} registros da collection '{collection_name}'...")

    # Paginacao por cursor de ID -----------------------------------------------
    last_id = -1
    max_iters = (total // MILVUS_QUERY_BATCH_SIZE) + 20  # cap de seguranca

    with tqdm(total=total, desc="Carregando registros", unit="reg") as pbar:
        for _ in range(max_iters):
            filter_expr = "" if last_id == -1 else f"id > {last_id}"

            try:
                batch = client.client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=["id", "person_id", "image_path", "embedding"],
                    limit=MILVUS_QUERY_BATCH_SIZE,
                )
            except Exception as e:
                print(f"\n✗ Erro ao buscar batch (cursor id>{last_id}): {e}")
                break

            if not batch:
                break

            all_records.extend(batch)
            new_last_id = max(r["id"] for r in batch)

            if new_last_id == last_id:
                print(f"\n⚠ Cursor de ID travado em {last_id}; encerrando paginacao.")
                break

            last_id = new_last_id
            pbar.update(len(batch))

            if len(batch) < MILVUS_QUERY_BATCH_SIZE:
                break

    print(f"✓ {len(all_records)} registros carregados")
    return all_records


def leave_one_out_search(
    client: MilvusClient,
    records: List[Dict],
    top_k: int = SEARCH_TOP_K,
) -> List[Dict]:
    """
    Para cada registro, busca os top_k vizinhos mais próximos,
    filtra o self-match por image_path e computa HR@1.

    Retorna lista de resultados por imagem.
    """
    results = []

    for record in tqdm(records, desc="Leave-one-out search", unit="img"):
        query_embedding = np.array(record["embedding"], dtype=np.float32)
        query_path = record["image_path"]
        query_person = record["person_id"]
        query_id = record["id"]

        # Buscar vizinhos
        search_results = client.search(
            query_embedding=query_embedding,
            top_k=top_k,
            output_fields=["person_id", "image_path"],
        )

        # Filtrar self-match pelo ID primário — mais robusto que image_path porque
        # o id vem de hit.id (sempre presente), enquanto image_path vem do entity
        # (pode ser None se o parse do pymilvus falhar, deixando o self-match passar).
        neighbors = [
            r for r in search_results
            if r.get("id") != query_id
        ]

        # Análise do resultado
        hr1_hit = False
        top1_person = None
        top1_path = None
        top1_similarity = None
        correct_rank = None  # Rank em que aparece o primeiro match correto

        if neighbors:
            top1 = neighbors[0]
            top1_person = top1.get("person_id")
            top1_path = top1.get("image_path")
            top1_similarity = top1.get("distance", 0.0)
            hr1_hit = (top1_person == query_person)

            # Encontrar rank do primeiro match correto
            for rank, neighbor in enumerate(neighbors[:MAX_RANK_TRACK], start=1):
                if neighbor.get("person_id") == query_person:
                    correct_rank = rank
                    break

        results.append({
            "query_id": query_id,
            "query_image_path": query_path,
            "query_person_id": query_person,
            "hr1_hit": hr1_hit,
            "top1_person_id": top1_person,
            "top1_image_path": top1_path,
            "top1_similarity": top1_similarity,
            "correct_rank": correct_rank,  # None = não encontrado no top_k
            "num_neighbors": len(neighbors),
        })

    return results


# =========================================================================
# Análise e agregação
# =========================================================================

def compute_per_identity_stats(results: List[Dict]) -> List[Dict]:
    """Agrega HR@1 por identidade."""
    identity_data = defaultdict(lambda: {"total": 0, "hits": 0, "failures": []})

    for r in results:
        pid = r["query_person_id"]
        identity_data[pid]["total"] += 1
        if r["hr1_hit"]:
            identity_data[pid]["hits"] += 1
        else:
            identity_data[pid]["failures"].append(r)

    stats = []
    for pid, data in identity_data.items():
        hr1_rate = data["hits"] / data["total"] if data["total"] > 0 else 0
        stats.append({
            "person_id": pid,
            "total_images": data["total"],
            "hr1_hits": data["hits"],
            "hr1_failures": data["total"] - data["hits"],
            "hr1_rate": hr1_rate,
            "failure_details": data["failures"],
        })

    # Ordenar por pior HR@1 rate, depois por mais falhas absolutas
    stats.sort(key=lambda x: (x["hr1_rate"], -x["hr1_failures"]))
    return stats


def compute_confusion_pairs(results: List[Dict]) -> List[Dict]:
    """Identifica pares de identidades mais confundidos."""
    confusion_counter = Counter()

    for r in results:
        if not r["hr1_hit"] and r["top1_person_id"]:
            # Normalizar a ordem do par para contar bidirecional
            pair = tuple(sorted([r["query_person_id"], r["top1_person_id"]]))
            confusion_counter[pair] += 1

    pairs = []
    for (pid_a, pid_b), count in confusion_counter.most_common():
        pairs.append({
            "person_a": pid_a,
            "person_b": pid_b,
            "confusion_count": count,
        })

    return pairs


# =========================================================================
# Exportação de relatórios
# =========================================================================

def save_per_image_csv(results: List[Dict], output_path: Path, only_failures: bool = False):
    """Salva relatório por imagem em CSV."""
    rows = results if not only_failures else [r for r in results if not r["hr1_hit"]]

    # Ordenar: failures primeiro, depois por similaridade do top-1 (desc)
    rows.sort(key=lambda x: (x["hr1_hit"], -(x["top1_similarity"] or 0)))

    fieldnames = [
        "query_image_path", "query_person_id", "hr1_hit",
        "top1_person_id", "top1_image_path", "top1_similarity",
        "correct_rank",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"✓ CSV per-image salvo: {output_path} ({len(rows)} registros)")


def save_per_identity_csv(identity_stats: List[Dict], output_path: Path):
    """Salva relatório por identidade em CSV."""
    fieldnames = [
        "person_id", "total_images", "hr1_hits", "hr1_failures", "hr1_rate",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stat in identity_stats:
            writer.writerow({k: stat[k] for k in fieldnames})

    print(f"✓ CSV per-identity salvo: {output_path} ({len(identity_stats)} identidades)")


def save_confusion_pairs_csv(pairs: List[Dict], output_path: Path):
    """Salva confusion pairs em CSV."""
    fieldnames = ["person_a", "person_b", "confusion_count"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)

    print(f"✓ CSV confusion pairs salvo: {output_path} ({len(pairs)} pares)")


def save_summary_json(
    model_name: str,
    collection_name: str,
    total_images: int,
    results: List[Dict],
    identity_stats: List[Dict],
    confusion_pairs: List[Dict],
    output_path: Path,
):
    """Salva sumário completo em JSON."""
    failures = [r for r in results if not r["hr1_hit"]]
    total_identities = len(identity_stats)
    identities_with_failures = sum(1 for s in identity_stats if s["hr1_failures"] > 0)

    # Distribuição de rank do match correto nos failures
    rank_distribution = Counter()
    for r in failures:
        rank = r["correct_rank"]
        rank_distribution[rank if rank else "not_in_top_k"] += 1

    summary = {
        "metadata": {
            "model": model_name,
            "collection": collection_name,
            "timestamp": datetime.now().isoformat(),
            "total_images": total_images,
            "total_identities": total_identities,
            "search_top_k": SEARCH_TOP_K,
        },
        "global_metrics": {
            "hr1": sum(1 for r in results if r["hr1_hit"]) / len(results) if results else 0,
            "hr1_hits": sum(1 for r in results if r["hr1_hit"]),
            "hr1_failures": len(failures),
            "failure_rate": len(failures) / len(results) if results else 0,
        },
        "identity_summary": {
            "total_identities": total_identities,
            "identities_with_failures": identities_with_failures,
            "identities_all_correct": total_identities - identities_with_failures,
            "worst_identities": [
                {
                    "person_id": s["person_id"],
                    "hr1_rate": round(s["hr1_rate"], 4),
                    "failures": s["hr1_failures"],
                    "total": s["total_images"],
                }
                for s in identity_stats[:20]  # Top 20 piores
                if s["hr1_failures"] > 0
            ],
        },
        "rank_distribution_of_failures": {
            str(k): v for k, v in sorted(
                rank_distribution.items(),
                key=lambda x: (isinstance(x[0], str), x[0])  # numéricos primeiro
            )
        },
        "top_confusion_pairs": [
            p for p in confusion_pairs[:20]
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON summary salvo: {output_path}")
    return summary


# =========================================================================
# Relatório visual HTML
# =========================================================================

THUMBNAIL_SIZE = (160, 160)


def _image_to_base64(image_path: str) -> Optional[str]:
    """Converte imagem para base64 data URI. Retorna None se não encontrar."""
    try:
        from PIL import Image
        import io

        path = Path(image_path)
        if not path.exists():
            return None

        img = Image.open(path).convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _placeholder_svg() -> str:
    """SVG placeholder quando a imagem não é encontrada."""
    return (
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            b'<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160">'
            b'<rect width="160" height="160" fill="#2a2a2a"/>'
            b'<text x="80" y="85" text-anchor="middle" fill="#888" font-size="13">'
            b'Not found</text></svg>'
        ).decode("utf-8")
    )


def generate_visual_report(
    model_name: str,
    collection_name: str,
    results: List[Dict],
    identity_stats: List[Dict],
    confusion_pairs: List[Dict],
    output_path: Path,
    max_items: int = 200,
):
    """
    Gera relatório HTML autocontido com imagens embutidas em base64.
    Mostra query vs top-1 incorreto lado a lado para cada falha.
    """
    total = len(results)
    hits = sum(1 for r in results if r["hr1_hit"])
    failures = [r for r in results if not r["hr1_hit"]]
    # Ordenar: maior similaridade primeiro (os erros mais "confiantes" são os mais perigosos)
    failures.sort(key=lambda x: -(x["top1_similarity"] or 0))
    failures_display = failures[:max_items]

    hr1_pct = hits / total * 100 if total else 0
    fail_pct = len(failures) / total * 100 if total else 0

    # Rank distribution
    rank_dist = Counter()
    for r in failures:
        rk = r["correct_rank"]
        rank_dist[rk if rk else "not_in_top_k"] += 1

    # Worst identities
    worst_ids = [s for s in identity_stats if s["hr1_failures"] > 0][:15]

    # Top confusion pairs
    top_conf = confusion_pairs[:10]

    # Pre-load images
    print(f"Gerando relatório visual ({len(failures_display)} falhas)...")
    image_cache = {}
    paths_needed = set()
    for r in failures_display:
        if r["query_image_path"]:
            paths_needed.add(r["query_image_path"])
        if r["top1_image_path"]:
            paths_needed.add(r["top1_image_path"])

    placeholder = _placeholder_svg()
    found_set: set = set()
    for p in tqdm(paths_needed, desc="Carregando imagens", unit="img"):
        b64 = _image_to_base64(p)
        if b64:
            image_cache[p] = b64
            found_set.add(p)
        else:
            image_cache[p] = placeholder

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Build HTML ---
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Failure Analysis — {html_module.escape(model_name)}</title>
<style>
  :root {{
    --bg: #0f0f0f; --surface: #1a1a1a; --border: #2a2a2a;
    --text: #e0e0e0; --text-dim: #888; --accent: #4a9eff;
    --success: #4caf50; --danger: #ef5350; --warn: #ff9800;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); padding: 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; font-weight: 600; margin: 32px 0 16px; color: var(--accent); }}
  .subtitle {{ color: var(--text-dim); font-size: 0.9rem; margin-bottom: 24px; }}

  /* Summary cards */
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 32px; }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; text-align: center;
  }}
  .stat-card .value {{ font-size: 1.8rem; font-weight: 700; }}
  .stat-card .label {{ font-size: 0.8rem; color: var(--text-dim); margin-top: 4px; }}
  .stat-card.success .value {{ color: var(--success); }}
  .stat-card.danger .value {{ color: var(--danger); }}

  /* Failure cards */
  .failure-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; margin-bottom: 12px;
    display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: start;
  }}
  .failure-card .img-col {{ text-align: center; }}
  .failure-card img {{
    width: 160px; height: 160px; object-fit: cover; border-radius: 6px;
    border: 2px solid var(--border); display: block; margin: 0 auto;
  }}
  .failure-card .img-col.wrong img {{ border-color: var(--danger); }}
  .failure-card .role-tag {{
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 6px;
  }}
  .role-query {{ background: #1a2e1a; color: var(--success); }}
  .role-wrong {{ background: #3a1a1a; color: var(--danger); }}
  .failure-card .person-label {{ font-size: 0.9rem; font-weight: 700; margin-top: 6px; }}
  .failure-card .img-label {{
    font-size: 0.72rem; color: var(--text-dim); margin-top: 4px;
    word-break: break-all; max-width: 180px; margin-left: auto; margin-right: auto;
  }}
  .failure-card .meta-line {{
    font-size: 0.75rem; color: var(--text-dim); margin-top: 3px;
  }}
  .failure-card .arrow-col {{
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    font-size: 0.8rem; color: var(--text-dim); padding-top: 40px;
  }}
  .failure-card .arrow-col .idx {{ color: var(--text-dim); font-size: 0.8rem; }}
  .failure-card .arrow-col .arrow {{ font-size: 1.8rem; color: var(--danger); line-height: 1; }}
  .failure-card .arrow-col .sim-label {{ font-size: 0.7rem; color: var(--text-dim); }}
  .failure-card .arrow-col .sim {{ font-size: 1.1rem; font-weight: 700; color: var(--warn); }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 600;
  }}
  .badge-rank {{ background: #1a3a5c; color: var(--accent); }}
  .badge-lost {{ background: #3a1a1a; color: var(--danger); }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 0.85rem; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-dim); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
  tr:hover {{ background: rgba(255,255,255,0.03); }}

  .bar {{ height: 6px; border-radius: 3px; background: var(--border); position: relative; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 3px; }}

  .truncated-note {{ color: var(--text-dim); font-size: 0.85rem; text-align: center; padding: 20px; }}
</style>
</head>
<body>

<h1>Failure Analysis — HR@1</h1>
<p class="subtitle">
  Modelo: <strong>{html_module.escape(model_name)}</strong> &middot;
  Collection: <strong>{html_module.escape(collection_name)}</strong> &middot;
  {html_module.escape(timestamp)}
</p>

<div class="summary-grid">
  <div class="stat-card"><div class="value">{total:,}</div><div class="label">Total Images</div></div>
  <div class="stat-card success"><div class="value">{hr1_pct:.1f}%</div><div class="label">HR@1</div></div>
  <div class="stat-card danger"><div class="value">{len(failures):,}</div><div class="label">Failures</div></div>
  <div class="stat-card"><div class="value">{len(identity_stats):,}</div><div class="label">Identities</div></div>
  <div class="stat-card danger"><div class="value">{len(worst_ids)}</div><div class="label">Identities w/ Failures</div></div>
</div>
""")

    # --- Rank distribution ---
    html_parts.append("<h2>Rank Distribution (Failures)</h2>")
    html_parts.append("""<p style="color:var(--text-dim);font-size:0.85rem;margin-bottom:12px;">
      Onde o match correto aparece quando o HR@1 falha.
      Rank 2 = &quot;quase acertou&quot;, not_in_top_k = caso grave.
    </p>""")
    html_parts.append("<table><tr><th>Rank</th><th>Count</th><th>%</th><th></th></tr>")
    for rk in sorted(rank_dist.keys(), key=lambda x: (isinstance(x, str), x)):
        cnt = rank_dist[rk]
        pct = cnt / len(failures) * 100 if failures else 0
        label = f"Rank {rk}" if isinstance(rk, int) else str(rk)
        color = "var(--accent)" if isinstance(rk, int) and rk <= 3 else "var(--danger)"
        html_parts.append(
            f'<tr><td>{label}</td><td>{cnt}</td><td>{pct:.1f}%</td>'
            f'<td><div class="bar" style="width:200px"><div class="bar-fill" '
            f'style="width:{pct}%;background:{color}"></div></div></td></tr>'
        )
    html_parts.append("</table>")

    # --- Worst identities ---
    if worst_ids:
        html_parts.append("<h2>Worst Identities</h2>")
        html_parts.append("<table><tr><th>Identity</th><th>HR@1</th><th>Hits</th><th>Failures</th><th>Total</th></tr>")
        for s in worst_ids:
            hr = s["hr1_rate"] * 100
            color = "var(--danger)" if hr < 50 else "var(--warn)" if hr < 80 else "var(--success)"
            html_parts.append(
                f'<tr><td>{html_module.escape(s["person_id"])}</td>'
                f'<td style="color:{color};font-weight:600">{hr:.1f}%</td>'
                f'<td>{s["hr1_hits"]}</td><td>{s["hr1_failures"]}</td><td>{s["total_images"]}</td></tr>'
            )
        html_parts.append("</table>")

    # --- Top confusion pairs ---
    if top_conf:
        html_parts.append("<h2>Top Confusion Pairs</h2>")
        html_parts.append("<table><tr><th>Person A</th><th>Person B</th><th>Confusions</th></tr>")
        for p in top_conf:
            html_parts.append(
                f'<tr><td>{html_module.escape(p["person_a"])}</td>'
                f'<td>{html_module.escape(p["person_b"])}</td>'
                f'<td>{p["confusion_count"]}</td></tr>'
            )
        html_parts.append("</table>")

    # --- Failure cards ---
    html_parts.append(
        f"<h2>Failure Details ({len(failures_display)} / {len(failures)})</h2>"
        f'<p style="color:var(--text-dim);font-size:0.85rem;margin-bottom:16px;">'
        f'Cada card mostra: <span style="color:var(--success)">QUERY</span> (imagem consultada) '
        f'&rarr; <span style="color:var(--danger)">ERRO</span> (imagem com que o modelo a confundiu, '
        f'pessoa diferente). A similaridade indica quanta confiança o modelo tinha na resposta errada.'
        f'</p>'
    )

    for i, r in enumerate(failures_display, 1):
        q_path = r["query_image_path"] or ""
        t_path = r["top1_image_path"] or ""

        q_img_found = q_path in found_set
        t_img_found = t_path in found_set

        q_img = image_cache.get(q_path, placeholder)
        t_img = image_cache.get(t_path, placeholder)

        q_name = Path(q_path).name if q_path else "imagem não encontrada"
        t_name = Path(t_path).name if t_path else "imagem não encontrada"
        q_person = html_module.escape(r["query_person_id"] or "?")
        t_person = html_module.escape(r["top1_person_id"] or "?")
        sim = f'{r["top1_similarity"]:.4f}' if r["top1_similarity"] is not None else "N/A"
        crank = r["correct_rank"]
        rank_badge = (
            f'<span class="badge badge-rank">match correto em rank {crank}</span>'
            if crank
            else '<span class="badge badge-lost">correto nao apareceu no top-k</span>'
        )
        not_found_q = '' if q_img_found else '<div class="meta-line" style="color:var(--danger)">arquivo nao encontrado</div>'
        not_found_t = '' if t_img_found else '<div class="meta-line" style="color:var(--danger)">arquivo nao encontrado</div>'

        html_parts.append(f"""
<div class="failure-card">
  <div class="img-col">
    <span class="role-tag role-query">QUERY</span>
    <img src="{q_img}" alt="query">
    <div class="person-label">{q_person}</div>
    <div class="img-label">{html_module.escape(q_name)}</div>
    {not_found_q}
  </div>
  <div class="arrow-col">
    <span class="idx">#{i}</span>
    <span class="arrow">&rarr;</span>
    <span class="sim-label">similaridade</span>
    <span class="sim">{sim}</span>
    {rank_badge}
  </div>
  <div class="img-col wrong">
    <span class="role-tag role-wrong">ERRO &mdash; confundido com</span>
    <img src="{t_img}" alt="top-1 errado">
    <div class="person-label" style="color:var(--danger)">{t_person}</div>
    <div class="img-label">{html_module.escape(t_name)}</div>
    {not_found_t}
  </div>
</div>""")

    if len(failures) > max_items:
        html_parts.append(
            f'<p class="truncated-note">Mostrando {max_items} de {len(failures)} falhas. '
            f'Use --max-visual para aumentar.</p>'
        )

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Relatório visual salvo: {output_path} ({size_mb:.1f} MB)")

def print_terminal_report(
    model_name: str,
    collection_name: str,
    results: List[Dict],
    identity_stats: List[Dict],
    confusion_pairs: List[Dict],
    top_n: Optional[int] = None,
):
    """Imprime relatório resumido no terminal."""
    total = len(results)
    hits = sum(1 for r in results if r["hr1_hit"])
    failures_list = [r for r in results if not r["hr1_hit"]]
    failures_list.sort(key=lambda x: -(x["top1_similarity"] or 0))

    print()
    print("=" * 80)
    print(f"  FAILURE ANALYSIS — HR@1")
    print(f"  Modelo: {model_name} | Collection: {collection_name}")
    print("=" * 80)

    print(f"\n  Total de imagens:     {total}")
    print(f"  HR@1 hits:            {hits} ({hits/total*100:.2f}%)")
    print(f"  HR@1 failures:        {len(failures_list)} ({len(failures_list)/total*100:.2f}%)")

    # Rank distribution dos failures
    rank_dist = Counter()
    for r in failures_list:
        rank = r["correct_rank"]
        rank_dist[rank if rank else ">10"] = rank_dist.get(rank if rank else ">10", 0) + 1

    print(f"\n  Distribuição de rank do match correto (nos failures):")
    for rank_key in sorted(rank_dist.keys(), key=lambda x: (isinstance(x, str), x)):
        count = rank_dist[rank_key]
        pct = count / len(failures_list) * 100 if failures_list else 0
        label = f"Rank {rank_key}" if isinstance(rank_key, int) else f"Rank {rank_key}"
        print(f"    {label}: {count} ({pct:.1f}%)")

    # Piores identidades
    worst_ids = [s for s in identity_stats if s["hr1_failures"] > 0]
    print(f"\n  Identidades com falhas: {len(worst_ids)} / {len(identity_stats)}")
    print(f"\n  Top identidades com pior HR@1:")
    print(f"  {'Identidade':<30} {'HR@1':>8} {'Hits':>6} {'Fails':>6} {'Total':>6}")
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    for s in worst_ids[:15]:
        print(
            f"  {s['person_id']:<30} "
            f"{s['hr1_rate']*100:>7.1f}% "
            f"{s['hr1_hits']:>6} "
            f"{s['hr1_failures']:>6} "
            f"{s['total_images']:>6}"
        )

    # Confusion pairs
    if confusion_pairs:
        print(f"\n  Top pares de confusão:")
        print(f"  {'Pessoa A':<25} {'Pessoa B':<25} {'Confusões':>10}")
        print(f"  {'-'*25} {'-'*25} {'-'*10}")
        for p in confusion_pairs[:10]:
            print(f"  {p['person_a']:<25} {p['person_b']:<25} {p['confusion_count']:>10}")

    # Top N piores imagens
    display_failures = failures_list[:top_n] if top_n else failures_list[:20]
    if display_failures:
        print(f"\n  Piores imagens (HR@1 = 0):")
        print(f"  {'Query Image':<40} {'Expected':<20} {'Got':<20} {'Sim':>6} {'CRank':>6}")
        print(f"  {'-'*40} {'-'*20} {'-'*20} {'-'*6} {'-'*6}")
        for r in display_failures:
            img_name = Path(r["query_image_path"]).name if r["query_image_path"] else "?"
            expected = r["query_person_id"] or "?"
            got = r["top1_person_id"] or "?"
            sim = f"{r['top1_similarity']:.4f}" if r["top1_similarity"] else "N/A"
            crank = str(r["correct_rank"]) if r["correct_rank"] else ">10"

            # Truncar nomes longos
            img_name = img_name[:38] if len(img_name) > 38 else img_name
            expected = expected[:18] if len(expected) > 18 else expected
            got = got[:18] if len(got) > 18 else got

            print(f"  {img_name:<40} {expected:<20} {got:<20} {sim:>6} {crank:>6}")

    print()
    print("=" * 80)


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    model_name = args.model
    collection_name = args.collection or Config.get_collection_name(model_name)
    db_path = args.db_path or Config.MILVUS_DB_PATH

    # Diretório de saída
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT_DIR / f"failure_analysis_{model_name}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  Failure Analysis — HR@1 per Image")
    print(f"{'='*80}")
    print(f"  Modelo:     {model_name}")
    print(f"  Collection: {collection_name}")
    print(f"  DB Path:    {db_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Search K:   {args.search_top_k}")
    print(f"{'='*80}\n")

    global SEARCH_TOP_K
    SEARCH_TOP_K = args.search_top_k

    # 1. Conectar ao Milvus
    client = MilvusClient(db_path=db_path, collection_name=collection_name)

    # 2. Buscar todos os registros
    records = fetch_all_records(client, collection_name)
    if not records:
        print("Nenhum registro encontrado. Abortando.")
        sys.exit(1)

    # 3. Leave-one-out search
    results = leave_one_out_search(client, records, top_k=SEARCH_TOP_K)

    # 4. Análise
    identity_stats = compute_per_identity_stats(results)
    confusion_pairs = compute_confusion_pairs(results)

    # 5. Relatório no terminal
    print_terminal_report(
        model_name, collection_name,
        results, identity_stats, confusion_pairs,
        top_n=args.top_n,
    )

    # 6. Salvar relatórios
    save_per_image_csv(
        results,
        output_dir / "per_image_results.csv",
        only_failures=args.only_failures,
    )
    save_per_identity_csv(identity_stats, output_dir / "per_identity_results.csv")
    save_confusion_pairs_csv(confusion_pairs, output_dir / "confusion_pairs.csv")
    summary = save_summary_json(
        model_name, collection_name,
        len(records), results, identity_stats, confusion_pairs,
        output_dir / "summary.json",
    )

    print(f"\n✓ Análise completa! Resultados em: {output_dir}")
    print(f"  - per_image_results.csv    → detalhes por imagem")
    print(f"  - per_identity_results.csv → agregado por identidade")
    print(f"  - confusion_pairs.csv      → pares confundidos")
    print(f"  - summary.json             → sumário geral")

    # 7. Relatório visual (opcional)
    if args.visual_report:
        generate_visual_report(
            model_name, collection_name,
            results, identity_stats, confusion_pairs,
            output_dir / "visual_report.html",
            max_items=args.max_visual,
        )
        print(f"  - visual_report.html       → relatório visual com imagens")


if __name__ == "__main__":
    main()