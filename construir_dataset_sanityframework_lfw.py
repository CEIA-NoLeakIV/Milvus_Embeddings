#!/usr/bin/env python3
"""
construir_dataset_sanityframework_lfw.py
=========================================
Constroi um dataset de validacao com as MESMAS identidades do LFW,
mas com imagens completamente diferentes.

SANITY FRAMEWORK (o que resolve o bug dos scripts anteriores)
-------------------------------------------------------------
Os scripts anteriores baixavam imagens via Bing sem nenhuma verificacao
de que a imagem realmente mostrava a pessoa correta. Isso causava:
  - Imagens de pessoas diferentes atribuidas a mesma identidade
  - Mesma imagem aparecendo em pastas de identidades diferentes

Este script usa TopoFR R100 (Glint360K) como verificador de identidade:
  1. Para cada identidade, calcula um embedding de referencia a partir
     das proprias imagens LFW (ground truth).
  2. Para cada imagem baixada, extrai o embedding TopoFR da face.
  3. So aceita a imagem se cosine_similarity(candidata, referencia)
     >= --sim-threshold (padrao: 0.30).

Isso garante que a imagem baixada e realmente da pessoa certa.

CRITERIOS DE QUALIDADE
----------------------
  1. Integridade do arquivo (PIL.verify)
  2. Resolucao minima >= --min-resolution (padrao: 112 px)
  3. Hash perceptual unico GLOBALMENTE (sem duplicatas entre identidades)
  4. Anti-leak: nao e imagem do LFW (Hamming distance > --hamming-threshold)
  5. Exatamente 1 face detectada pelo SCRFD (rejeita 0_faces e multi_faces)
  6. Sanity: similaridade TopoFR >= --sim-threshold com a referencia LFW

DIVERSIDADE
-----------
10 queries diferentes por identidade para capturar variacao de:
  - Pose: frontal, perfil, 3/4
  - Iluminacao: natural, artificial, studio, outdoor
  - Contexto: entrevista, evento, discurso, cerimonia
  - Idade: fotos recentes e antigas

Uso:
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --min-lfw-images 2
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --images-per-id 10
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --sim-threshold 0.25
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --start-from 500
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --max-identities 100
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --dry-run
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --check-watermarks
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --check-face-count

Setup:
  pip install icrawler Pillow tqdm numpy torch torchvision
  # Para watermark (opcional):
  pip install ollama
  ollama pull gemma3
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from icrawler.builtin import BingImageCrawler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LFW_DIR = SCRIPT_DIR / "lfw"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset_ext_val_lfw"
TOPOFR_WEIGHTS = SCRIPT_DIR / "models" / "weights" / "Glint360K_R100_TopoFR_9760.pt"

# Garantir que o diretorio raiz do projeto esteja no sys.path
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
VALID_EXT = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGES_PER_ID = 5
DEFAULT_MIN_RES = 112           # px — minimo para modelos de face recognition
DEFAULT_SIM_THRESHOLD = 0.30    # similaridade coseno TopoFR: mesma pessoa >= 0.3
DEFAULT_HAMMING_THRESHOLD = 4   # distancia Hamming para detectar near-duplicatas do LFW
DEFAULT_MIN_LFW_IMAGES = 1      # minimo de imgs LFW para incluir identidade
MAX_LFW_REF_IMAGES = 10         # maximo de imgs LFW para construir referencia

# Queries diversificadas: pose, iluminacao, idade, contexto
# Cada query usa filtro Bing type=photo (exclui ilustracoes, clipart, GIFs)
QUERIES = [
    "{name} face portrait photograph",
    "{name} side profile photograph",
    "{name} press conference photograph closeup",
    "{name} candid event photograph face",
    "{name} young photograph",
    "{name} official photograph portrait",
    "{name} speech photograph face",
    "{name} interview television photograph",
    "{name} award ceremony photograph",
    "{name} recent photograph face closeup",
]

BING_FILTERS = {"type": "photo"}

WATERMARK_PROMPT = (
    "Analyze this image carefully and answer YES if ANY of the following is true:\n"
    "1. The image contains a visible watermark, logo, stock photo text, copyright "
    "text, or semi-transparent branding (alamy, shutterstock, getty, dreamstime, etc.)\n"
    "2. The image is a drawing, painting, illustration, caricature, cartoon, "
    "sketch, digital art, or any non-photographic artwork\n"
    "Answer ONLY with YES or NO. Nothing else."
)

# ---------------------------------------------------------------------------
# Logging — silencia icrawler e urllib3
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.CRITICAL)
log = logging.getLogger("sanity_fw")
log.setLevel(logging.INFO)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_ch)
log.propagate = False

for _noisy in ["icrawler", "icrawler.crawler", "icrawler.feeder",
               "icrawler.parser", "icrawler.downloader", "urllib3",
               "urllib3.connectionpool", "PIL.Image"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# ===========================================================================
# TopoFR
# ===========================================================================

def init_topofr_model():
    """
    Inicializa o modelo TopoFR R100 Glint360K.
    Usa preprocessing.py do projeto para deteccao de face e extracao de embedding.
    """
    if not TOPOFR_WEIGHTS.exists():
        print(f"ERRO: peso do modelo nao encontrado: {TOPOFR_WEIGHTS}")
        print(f"  Esperado em: {TOPOFR_WEIGHTS}")
        sys.exit(1)

    try:
        from models.topofr_model import TopoFRModel
        model = TopoFRModel(weight_path=str(TOPOFR_WEIGHTS), use_tta=False)
        log.info(f"TopoFR R100 Glint360K inicializado: {TOPOFR_WEIGHTS.name}")
        return model
    except Exception as e:
        print(f"ERRO ao inicializar TopoFR: {e}")
        sys.exit(1)


def count_faces(path: Path) -> Tuple[int, str]:
    """
    Conta faces detectadas via SCRFD em uma imagem.
    Retorna (num_faces, status).
    status: 'ok' | 'erro:<msg>'.
    Usa o mesmo singleton de detector que preprocessing.py, garantindo
    consistencia com o pipeline principal.
    """
    try:
        import cv2
        from utils.face_detection import _get_detector
    except ImportError as e:
        return -1, f"erro:detector_import:{e}"

    try:
        img = cv2.imread(str(path))
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        faces = _get_detector().detect(img)
        return len(faces), "ok"
    except Exception as e:
        return -1, f"erro:{e}"


def _extract_embedding_topofr(model, path: Path) -> Tuple[Optional[np.ndarray], str]:
    """
    Detecta face e extrai embedding TopoFR L2-normalizado.
    Retorna (embedding, status).
    status pode ser: 'ok', '0_faces', 'multi_faces', 'embedding_invalido', 'erro:<msg>'.

    Pre-checa o numero de faces antes do embedding: aceita SOMENTE quando ha
    exatamente 1 face. Imagens com 0 ou >1 faces sao rejeitadas.
    """
    try:
        from preprocessing import NoFaceDetectedError
    except ImportError as e:
        return None, f"erro:preprocessing_import:{e}"

    n, status = count_faces(path)
    if status != "ok":
        return None, status
    if n == 0:
        return None, "0_faces"
    if n > 1:
        return None, f"multi_faces:{n}"

    try:
        pil_img = Image.open(path).convert("RGB")
        emb = model.extract_embedding_from_pil(pil_img)
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return None, "embedding_invalido"
        return emb / norm, "ok"
    except Exception as e:
        if isinstance(e, NoFaceDetectedError):
            return None, "0_faces"
        return None, f"erro:{e}"


def build_reference_embedding(
    model, lfw_dir: Path, identity: str
) -> Optional[np.ndarray]:
    """
    Constroi embedding de referencia para uma identidade usando imagens LFW.
    Retorna a media L2-normalizada dos embeddings validos.
    Retorna None se nenhuma face valida for encontrada nas imagens LFW.

    A media de multiplas imagens torna a referencia mais robusta a variacao
    de pose/iluminacao dentro do LFW.
    """
    try:
        from preprocessing import NoFaceDetectedError
    except ImportError:
        return None

    id_dir = lfw_dir / identity
    if not id_dir.exists():
        return None

    imgs = sorted([
        f for f in id_dir.iterdir()
        if f.suffix.lower() in VALID_EXT and "Zone" not in f.name
    ])[:MAX_LFW_REF_IMAGES]

    embeddings = []
    for img_path in imgs:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            emb = model.extract_embedding_from_pil(pil_img)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                embeddings.append(emb / norm)
        except NoFaceDetectedError:
            continue
        except Exception:
            continue

    if not embeddings:
        return None

    avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 1e-6 else None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade coseno entre vetores L2-normalizados."""
    return float(np.dot(a, b))


# ===========================================================================
# Hash perceptual
# ===========================================================================

def phash(img: Image.Image, size: int = 8) -> str:
    """Average hash perceptual — 64 bits como string binaria."""
    gray = img.resize((size, size), Image.Resampling.LANCZOS).convert("L")
    px = list(gray.getdata())
    avg = sum(px) / len(px)
    return "".join("1" if p > avg else "0" for p in px)


def hamming(h1: str, h2: str) -> int:
    """Distancia de Hamming entre dois hashes de mesmo comprimento."""
    return sum(b1 != b2 for b1, b2 in zip(h1, h2))


def compute_lfw_hashes(lfw_dir: Path, identity: str) -> Set[str]:
    """Hashes perceptuais de todas as imagens LFW de uma identidade."""
    hashes: Set[str] = set()
    d = lfw_dir / identity
    if not d.exists():
        return hashes
    for f in d.iterdir():
        if f.suffix.lower() in VALID_EXT and "Zone" not in f.name:
            try:
                hashes.add(phash(Image.open(f)))
            except Exception:
                pass
    return hashes


# ===========================================================================
# Validacao de imagem
# ===========================================================================

def validate_image(
    path: Path,
    global_hashes: Set[str],
    lfw_hashes: Set[str],
    reference_emb: np.ndarray,
    model,
    min_res: int,
    sim_threshold: float,
    hamming_threshold: int,
) -> Tuple[bool, str, Optional[str]]:
    """
    Valida uma imagem candidata contra todos os criterios.
    Retorna (aceita, motivo_rejeicao_ou_ok, hash_perceptual).

    Ordem dos criterios (do mais barato ao mais caro):
      1. Integridade / extensao
      2. Resolucao
      3. Hash global (duplicata cross-identity)
      4. Anti-leak LFW (Hamming)
      5. Deteccao de face (RetinaFace via preprocessing.py)
      6. Sanity TopoFR (verificacao de identidade)
    """
    # --- 1. Integridade ---
    try:
        im = Image.open(path)
        im.verify()
        im = Image.open(path).convert("RGB")
    except Exception as e:
        return False, f"corrompida:{e}", None

    if path.suffix.lower() not in VALID_EXT:
        return False, "extensao_invalida", None

    # --- 2. Resolucao ---
    w, h = im.size
    if w < min_res or h < min_res:
        return False, f"resolucao_{w}x{h}", None

    # --- 3. Hash global (sem duplicatas entre identidades) ---
    ph = phash(im)
    if ph in global_hashes:
        return False, "duplicata_global", None

    # --- 4. Anti-leak LFW ---
    for lfw_h in lfw_hashes:
        if hamming(ph, lfw_h) <= hamming_threshold:
            return False, "similar_lfw", None

    # --- 5. Deteccao de face + 6. Sanity TopoFR ---
    emb, status = _extract_embedding_topofr(model, path)
    if status != "ok":
        return False, status, None

    sim = cosine_sim(emb, reference_emb)
    if sim < sim_threshold:
        return False, f"sim_{sim:.3f}<{sim_threshold}", None

    return True, f"ok(sim={sim:.3f})", ph


# ===========================================================================
# Coleta para uma identidade
# ===========================================================================

def collect_identity(
    identity: str,
    target_dir: Path,
    model,
    reference_emb: np.ndarray,
    lfw_hashes: Set[str],
    global_hashes: Set[str],
    target_count: int,
    min_res: int,
    sim_threshold: float,
    hamming_threshold: int,
) -> Dict:
    """
    Coleta e valida imagens para uma identidade.
    Atualiza global_hashes in-place quando aceita uma imagem.
    Retorna dict de estatisticas.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    name = identity.replace("_", " ")

    # --- Checar existentes ---
    existing = sorted(target_dir.glob("web_*"))
    if len(existing) >= target_count:
        for f in existing:
            try:
                global_hashes.add(phash(Image.open(f)))
            except Exception:
                pass
        return {
            "status": "skipped",
            "total": len(existing),
            "new": 0,
            "errors": [],
        }

    # Carregar hashes de imagens ja existentes para esta identidade
    for f in existing:
        try:
            global_hashes.add(phash(Image.open(f)))
        except Exception:
            pass

    collected = len(existing)
    # Naming pode ter gaps (ex: web_001, web_003 se web_002 falhou recrawl).
    # next_idx baseia-se no maior indice existente para nao sobrescrever.
    existing_indices = []
    for f in existing:
        try:
            existing_indices.append(int(f.stem.split("_")[-1]))
        except (ValueError, IndexError):
            continue
    next_idx = (max(existing_indices) + 1) if existing_indices else 1
    errors: List[str] = []

    tmp = target_dir / "_tmp"
    tmp.mkdir(exist_ok=True)

    queries = [q.format(name=name) for q in QUERIES]

    for query in queries:
        if collected >= target_count:
            break

        # Limpar tmp antes de cada query
        for f in tmp.iterdir():
            try:
                f.unlink()
            except Exception:
                pass

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(tmp)},
                log_level=logging.CRITICAL,
            )
            crawler.crawl(keyword=query, max_num=4, filters=BING_FILTERS)
        except Exception as e:
            errors.append(f"crawl|{query[:30]}|{e}")
            continue

        for dl in sorted(tmp.glob("*")):
            if collected >= target_count:
                break
            if dl.suffix.lower() not in VALID_EXT | {".gif", ".bmp", ".webp"}:
                continue

            try:
                im = Image.open(dl).convert("RGB")
                out = target_dir / f"web_{str(next_idx).zfill(3)}.jpg"
                im.save(out, "JPEG", quality=95)
            except Exception as e:
                errors.append(f"save|{e}")
                continue

            ok, reason, ph = validate_image(
                out,
                global_hashes,
                lfw_hashes,
                reference_emb,
                model,
                min_res,
                sim_threshold,
                hamming_threshold,
            )

            if ok:
                global_hashes.add(ph)
                collected += 1
                next_idx += 1
                log.debug(f"  ACEITA {out.name}: {reason}")
            else:
                out.unlink(missing_ok=True)
                errors.append(f"reject|{reason}|{query[:30]}")
                log.debug(f"  REJEIT {identity}/{next_idx}: {reason}")

        time.sleep(0.3)

    # Limpar tmp
    try:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    except Exception:
        pass

    status = (
        "ok" if collected >= target_count else
        "parcial" if collected > 0 else
        "falha"
    )

    return {
        "status": status,
        "total": collected,
        "new": collected - len(existing),
        "errors": errors,
    }


# ===========================================================================
# Watermark check via Ollama (opcional)
# ===========================================================================

def init_ollama(model: str, host: str):
    try:
        import ollama as _ol
        client = _ol.Client(host=host)
        names = [m.model for m in client.list().models]
        if not any(model in n for n in names):
            print(f"ERRO: modelo '{model}' nao encontrado no Ollama.")
            print(f"  Baixe com: ollama pull {model}")
            sys.exit(1)
        return client
    except ImportError:
        print("ERRO: pip install ollama")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO: Ollama nao disponivel em {host}: {e}")
        print("  Inicie com: ollama serve")
        sys.exit(1)


def check_watermark(client, path: Path, model: str) -> Tuple[bool, str]:
    try:
        resp = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": WATERMARK_PROMPT,
                "images": [str(path)],
            }],
        )
        answer = resp.message.content.strip().upper()
        return "YES" in answer, answer
    except Exception as e:
        return False, f"ERROR:{e}"


def recrawl_single(
    identity: str,
    target_dir: Path,
    filename: str,
    model,
    reference_emb: np.ndarray,
    lfw_hashes: Set[str],
    global_hashes: Set[str],
    min_res: int,
    sim_threshold: float,
    hamming_threshold: int,
    max_attempts: int = 3,
) -> bool:
    """Tenta substituir uma imagem especifica. Retorna True se substituiu."""
    name = identity.replace("_", " ")
    tmp = target_dir / "_tmp_wm"
    tmp.mkdir(exist_ok=True)

    for attempt in range(max_attempts):
        query = QUERIES[attempt % len(QUERIES)].format(name=name)

        for f in tmp.iterdir():
            try:
                f.unlink()
            except Exception:
                pass

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(tmp)},
                log_level=logging.CRITICAL,
            )
            crawler.crawl(keyword=query, max_num=5, filters=BING_FILTERS)
        except Exception:
            continue

        for dl in sorted(tmp.glob("*")):
            if dl.suffix.lower() not in VALID_EXT | {".gif", ".bmp", ".webp"}:
                continue
            try:
                im = Image.open(dl).convert("RGB")
                cand = target_dir / filename
                im.save(cand, "JPEG", quality=95)
            except Exception:
                continue

            ok, _, ph = validate_image(
                cand,
                global_hashes,
                lfw_hashes,
                reference_emb,
                model,
                min_res,
                sim_threshold,
                hamming_threshold,
            )
            if ok:
                global_hashes.add(ph)
                try:
                    for f in tmp.iterdir():
                        f.unlink()
                    tmp.rmdir()
                except Exception:
                    pass
                return True
            else:
                cand.unlink(missing_ok=True)

        time.sleep(0.3)

    try:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    except Exception:
        pass
    return False


# ===========================================================================
# Verificacao retroativa de numero de faces
# ===========================================================================

def verify_and_replace_face_count(
    identities: List[Tuple[str, int]],
    output_dir: Path,
    lfw_dir: Path,
    model,
    global_hashes: Set[str],
    args,
) -> Dict:
    """
    Percorre todas as imagens ja coletadas e verifica se cada uma contem
    exatamente 1 face. As que tiverem 0 ou >1 sao removidas e substituidas
    via recrawl_single (que reaplica TODOS os filtros, incluindo o novo
    filtro de 1-face).

    Retorna dict com estatisticas e a lista de itens processados.
    """
    all_imgs = [
        (name, f)
        for name, _ in identities
        for f in sorted((output_dir / name).glob("web_*"))
        if (output_dir / name).exists()
    ]

    print(f"\n  Total de imagens a verificar: {len(all_imgs)}")

    stats = {"ok_1_face": 0, "no_face": 0, "multi_face": 0, "error": 0}
    to_replace: List[Dict] = []

    for name, img_path in tqdm(all_imgs, desc="Contando faces", unit="img"):
        n, status = count_faces(img_path)
        if status != "ok":
            stats["error"] += 1
            continue
        if n == 1:
            stats["ok_1_face"] += 1
        elif n == 0:
            stats["no_face"] += 1
            to_replace.append({
                "identity": name,
                "file": img_path.name,
                "path": str(img_path),
                "num_faces": 0,
                "issue": "no_face",
            })
        else:
            stats["multi_face"] += 1
            to_replace.append({
                "identity": name,
                "file": img_path.name,
                "path": str(img_path),
                "num_faces": n,
                "issue": "multi_face",
            })

    print(
        f"\n  1 face (ok):       {stats['ok_1_face']}"
        f"\n  Sem face:          {stats['no_face']}"
        f"\n  Multiplas faces:   {stats['multi_face']}"
        f"\n  Erros de leitura:  {stats['error']}"
    )

    if not to_replace:
        print("\n  Nenhuma imagem precisa ser substituida.")
        return {
            "stats": {**stats, "replaced": 0, "failed": 0, "to_replace_total": 0},
            "items": [],
        }

    # Indexa hashes existentes para deduplicacao cross-identity durante recrawl
    print("\n  Indexando hashes das imagens existentes...")
    for _, img_path in tqdm(all_imgs, desc="phash existentes", unit="img"):
        try:
            global_hashes.add(phash(Image.open(img_path)))
        except Exception:
            pass

    print(f"\n  Substituindo {len(to_replace)} imagens (0 ou >1 faces)...")
    replaced = 0
    failed = 0

    for item in tqdm(to_replace, desc="Substituindo", unit="img"):
        ip = Path(item["path"])
        td = ip.parent
        name = item["identity"]

        ref = build_reference_embedding(model, lfw_dir, name)
        if ref is None:
            failed += 1
            item["resolved"] = False
            item["resolution"] = "no_ref_embedding"
            continue

        lfw_h = compute_lfw_hashes(lfw_dir, name)
        ip.unlink(missing_ok=True)

        if recrawl_single(
            name, td, item["file"],
            model, ref, lfw_h,
            global_hashes,
            args.min_resolution,
            args.sim_threshold,
            args.hamming_threshold,
        ):
            replaced += 1
            item["resolved"] = True
            item["resolution"] = "replaced"
        else:
            failed += 1
            item["resolved"] = False
            item["resolution"] = "recrawl_failed"

        time.sleep(0.3)

    print(f"\n  Substituidas: {replaced}  Falhas: {failed}")

    return {
        "stats": {
            **stats,
            "replaced": replaced,
            "failed": failed,
            "to_replace_total": len(to_replace),
        },
        "items": to_replace,
    }


# ===========================================================================
# Carregamento de identidades LFW
# ===========================================================================

def load_lfw_identities(
    lfw_dir: Path,
    min_images: int = DEFAULT_MIN_LFW_IMAGES,
) -> List[Tuple[str, int]]:
    """
    Retorna lista de (identity_name, num_lfw_images) para todas as
    identidades do LFW com >= min_images fotos.
    Ordenada decrescentemente por numero de imagens.
    """
    if not lfw_dir.exists():
        print(f"ERRO: diretorio LFW nao encontrado: {lfw_dir}")
        sys.exit(1)

    result = []
    for d in lfw_dir.iterdir():
        if not d.is_dir():
            continue
        n = len([
            f for f in d.iterdir()
            if f.suffix.lower() in VALID_EXT and "Zone" not in f.name
        ])
        if n >= min_images:
            result.append((d.name, n))

    result.sort(key=lambda x: -x[1])  # mais imagens primeiro (mais confiaveis)
    log.info(f"LFW: {len(result)} identidades com >= {min_images} imagem(ns)")
    return result


# ===========================================================================
# Relatorio
# ===========================================================================

def generate_report(
    identities: List[Tuple[str, int]],
    output_dir: Path,
    crawl_log: Dict,
    elapsed,
    args,
) -> str:
    total_ids = len(identities)
    counts = {"ok": 0, "parcial": 0, "falha": 0, "skipped": 0}
    ref_failed = 0

    for name, _ in identities:
        entry = crawl_log.get(name, {})
        st = entry.get("status", "?")
        counts[st] = counts.get(st, 0) + 1
        if entry.get("ref_failed"):
            ref_failed += 1

    total_imgs = sum(
        len(list((output_dir / name).glob("web_*")))
        for name, _ in identities
        if (output_dir / name).exists()
    )

    size_mb = sum(
        f.stat().st_size
        for name, _ in identities
        for f in (output_dir / name).glob("web_*")
        if f.exists()
    ) / (1024 * 1024)

    incomplete = [
        (name, len(list((output_dir / name).glob("web_*"))))
        for name, _ in identities
        if (output_dir / name).exists()
        and len(list((output_dir / name).glob("web_*"))) < args.images_per_id
    ]

    lines = [
        "",
        "=" * 72,
        "  RELATORIO — Dataset de Validacao (Sanity Framework LFW + TopoFR)",
        "=" * 72,
        f"  Data:                  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Tempo total:           {elapsed}",
        f"  Diretorio:             {output_dir}",
        "",
        "  --- Configuracao ---",
        f"  LFW dir:               {args.lfw_dir}",
        f"  Modelo:                TopoFR R100 Glint360K",
        f"  Min LFW imgs/id:       {args.min_lfw_images}",
        f"  Imgs alvo/identidade:  {args.images_per_id}",
        f"  Threshold TopoFR:      {args.sim_threshold}",
        f"  Resolucao minima:      {args.min_resolution}px",
        f"  Hamming threshold:     {args.hamming_threshold}",
        "",
        "  --- Visao Geral ---",
        f"  Identidades LFW:       {total_ids}",
        f"  Identidades sem ref:   {ref_failed}  (sem face valida no LFW)",
        f"  Imagens coletadas:     {total_imgs}",
        f"  Tamanho do dataset:    {size_mb:.1f} MB",
        "",
        "  --- Status de Coleta ---",
        f"  Completas (>= {args.images_per_id} imgs): {counts.get('ok', 0)}",
        f"  Parciais (< {args.images_per_id} imgs):  {counts.get('parcial', 0)}",
        f"  Falha (0 imgs):        {counts.get('falha', 0)}",
        f"  Puladas (ja ok):       {counts.get('skipped', 0)}",
    ]

    if incomplete:
        lines.extend([
            "",
            f"  --- Identidades Incompletas ({len(incomplete)}) ---",
        ])
        for name, c in sorted(incomplete, key=lambda x: x[1])[:30]:
            lines.append(f"    {name:<45s} {c} imgs")
        if len(incomplete) > 30:
            lines.append(f"    ... e mais {len(incomplete) - 30}")

    lines.append("=" * 72)
    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Constroi dataset com identidades do LFW + sanity framework TopoFR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --min-lfw-images 5
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --dry-run
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --start-from 500
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --check-watermarks
  python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --check-face-count
        """,
    )

    p.add_argument(
        "--lfw-dir", type=str, default=str(DEFAULT_LFW_DIR),
        help=f"Diretorio com o dataset LFW (default: {DEFAULT_LFW_DIR})",
    )
    p.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Diretorio de saida (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--min-lfw-images", type=int, default=DEFAULT_MIN_LFW_IMAGES,
        help=(
            f"Minimo de imagens no LFW para incluir identidade (default: {DEFAULT_MIN_LFW_IMAGES}). "
            "Use >= 2 para referencias mais confiaveis."
        ),
    )
    p.add_argument(
        "--images-per-id", type=int, default=DEFAULT_IMAGES_PER_ID,
        help=f"Imagens a coletar por identidade (default: {DEFAULT_IMAGES_PER_ID})",
    )
    p.add_argument(
        "--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD,
        help=(
            f"Threshold de similaridade coseno TopoFR (default: {DEFAULT_SIM_THRESHOLD}). "
            "Mesma pessoa tipicamente >= 0.30, pessoas diferentes tipicamente <= 0.10."
        ),
    )
    p.add_argument(
        "--min-resolution", type=int, default=DEFAULT_MIN_RES,
        help=f"Resolucao minima em pixels (default: {DEFAULT_MIN_RES})",
    )
    p.add_argument(
        "--hamming-threshold", type=int, default=DEFAULT_HAMMING_THRESHOLD,
        help=(
            f"Distancia Hamming maxima para considerar imagem igual ao LFW "
            f"(default: {DEFAULT_HAMMING_THRESHOLD}). 0 = identico exato, 4 = near-duplicate."
        ),
    )
    p.add_argument(
        "--max-identities", type=int, default=None,
        help="Limitar numero de identidades (util para testes)",
    )
    p.add_argument(
        "--start-from", type=int, default=0,
        help="Retomar a partir do indice N (default: 0)",
    )
    p.add_argument(
        "--check-watermarks", action="store_true",
        help="Ativar verificacao de marca d'agua via Ollama VLM apos coleta",
    )
    p.add_argument(
        "--check-face-count", action="store_true",
        help=(
            "Modo verificacao: percorre todas as imagens ja coletadas, "
            "rejeita as que tem 0 ou >1 faces e tenta substitui-las. "
            "PULA a coleta principal e o watermark check."
        ),
    )
    p.add_argument(
        "--ollama-model", type=str, default="gemma3",
        help="Modelo Ollama para watermark check (default: gemma3)",
    )
    p.add_argument(
        "--ollama-host", type=str, default="http://localhost:11434",
        help="Host do Ollama (default: http://localhost:11434)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Lista identidades e configuracao sem coletar",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Logging detalhado (mostra motivos de rejeicao por imagem)",
    )

    args = p.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    lfw_dir = Path(args.lfw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_path = output_dir / "log_coleta.json"
    report_path = output_dir / "relatorio_coleta.txt"
    csv_path = output_dir / "identidades.csv"

    # -----------------------------------------------------------------------
    # 1. Carregar identidades do LFW
    # -----------------------------------------------------------------------
    identities = load_lfw_identities(lfw_dir, min_images=args.min_lfw_images)

    if args.max_identities:
        identities = identities[: args.max_identities]

    print("=" * 72)
    print("  DATASET DE VALIDACAO — Sanity Framework LFW + TopoFR")
    print("=" * 72)
    print(f"  LFW dir:               {lfw_dir}")
    print(f"  Output dir:            {output_dir}")
    print(f"  Modelo:                TopoFR R100 Glint360K")
    print(f"  Identidades:           {len(identities)}")
    print(f"  Imgs alvo / id:        {args.images_per_id}")
    print(f"  Meta total:            ~{len(identities) * args.images_per_id} imagens")
    print(f"  Threshold TopoFR:      {args.sim_threshold}")
    print(f"  Resolucao minima:      {args.min_resolution}px")
    print(f"  Hamming anti-leak:     {args.hamming_threshold}")
    print(f"  Watermark check:       {'Sim' if args.check_watermarks else 'Nao'}")
    print(f"  Face-count check:      {'Sim (modo verificacao)' if args.check_face_count else 'Nao'}")
    print(f"  Retomada a partir de:  #{args.start_from}")
    print("=" * 72)

    if args.dry_run:
        print(f"\n[DRY RUN] Primeiras 50 identidades:")
        for i, (name, n_lfw) in enumerate(identities[:50]):
            print(f"  {i+1:5d}. {name:<45s} ({n_lfw} imgs LFW)")
        if len(identities) > 50:
            print(f"  ... e mais {len(identities) - 50} identidades")
        print(f"\n  Total: {len(identities)} identidades")
        return

    # -----------------------------------------------------------------------
    # 2. Criar estrutura de saida
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 3. Inicializar TopoFR
    # -----------------------------------------------------------------------
    model = init_topofr_model()

    # -----------------------------------------------------------------------
    # 4. Carregar log existente (resumabilidade)
    # -----------------------------------------------------------------------
    crawl_log: Dict[str, Dict] = {}
    if log_path.exists():
        try:
            with open(log_path) as f:
                crawl_log = json.load(f)
            log.info(f"Log existente carregado: {len(crawl_log)} entradas")
        except Exception:
            crawl_log = {}

    # -----------------------------------------------------------------------
    # 5. Construir hash global a partir de imagens ja coletadas
    #    (necessario para deduplicacao cross-identity ao retomar)
    # -----------------------------------------------------------------------
    global_hashes: Set[str] = set()
    log.info("Reconstruindo hash global de imagens existentes...")
    already_done = [
        name for name, _ in identities
        if (output_dir / name).exists()
        and len(list((output_dir / name).glob("web_*"))) >= args.images_per_id
    ]
    for name in tqdm(already_done, desc="Carregando hashes existentes", unit="id",
                     disable=not already_done):
        for f in (output_dir / name).glob("web_*"):
            try:
                global_hashes.add(phash(Image.open(f)))
            except Exception:
                pass

    log.info(f"Hash global: {len(global_hashes)} imagens pre-existentes indexadas")

    # -----------------------------------------------------------------------
    # 5b. Modo verificacao retroativa de num faces (--check-face-count)
    #     Pula a coleta principal e o watermark check.
    # -----------------------------------------------------------------------
    if args.check_face_count:
        print("\n" + "=" * 72)
        print("  [Modo verificacao] Conferindo num de faces nas imgs coletadas...")
        print("=" * 72)

        t0 = datetime.now()
        result = verify_and_replace_face_count(
            identities=identities,
            output_dir=output_dir,
            lfw_dir=lfw_dir,
            model=model,
            global_hashes=global_hashes,
            args=args,
        )
        elapsed = datetime.now() - t0

        fc_log_path = output_dir / "log_face_count.json"
        with open(fc_log_path, "w", encoding="utf-8") as f:
            json.dump({
                "elapsed_seconds": elapsed.total_seconds(),
                "elapsed_hms": str(elapsed),
                "timestamp": datetime.now().isoformat(),
                **result,
            }, f, indent=2, ensure_ascii=False)
        log.info(f"Log de face-count salvo: {fc_log_path}")

        s = result["stats"]
        print("\n" + "=" * 72)
        print("  RESUMO — Verificacao de num de faces")
        print("=" * 72)
        print(f"  Tempo:               {elapsed}")
        print(f"  1 face (ok):         {s['ok_1_face']}")
        print(f"  Sem face:            {s['no_face']}")
        print(f"  Multiplas faces:    {s['multi_face']}")
        print(f"  Erros de leitura:    {s['error']}")
        print(f"  Tentativas de subst: {s['to_replace_total']}")
        print(f"    Substituidas:      {s['replaced']}")
        print(f"    Falhas:            {s['failed']}")
        print("=" * 72)
        return

    # -----------------------------------------------------------------------
    # 6. Coleta principal
    # -----------------------------------------------------------------------
    print(f"\n[Fase 1/2] Coletando imagens com validacao TopoFR...")
    print(f"  Cada imagem passa por: integridade -> resolucao -> hash global ->")
    print(f"  anti-leak LFW -> deteccao de face -> similaridade TopoFR >= {args.sim_threshold}\n")

    t0 = datetime.now()
    cnt = {"ok": 0, "parcial": 0, "falha": 0, "skipped": 0, "no_ref": 0}

    subset = identities[args.start_from:]

    pbar = tqdm(
        subset,
        desc="Coletando",
        unit="id",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for identity, n_lfw in pbar:
        # --- Construir referencia TopoFR desta identidade ---
        reference_emb = build_reference_embedding(model, lfw_dir, identity)

        if reference_emb is None:
            log.debug(f"Sem referencia valida para {identity} ({n_lfw} imgs LFW) — pulando")
            cnt["no_ref"] += 1
            crawl_log[identity] = {
                "status": "falha",
                "total": 0,
                "new": 0,
                "errors": ["no_valid_reference"],
                "ref_failed": True,
                "n_lfw": n_lfw,
                "ts": datetime.now().isoformat(),
            }
            continue

        # --- Hashes LFW desta identidade (anti-leak) ---
        lfw_hashes = compute_lfw_hashes(lfw_dir, identity)

        # --- Coletar ---
        result = collect_identity(
            identity=identity,
            target_dir=output_dir / identity,
            model=model,
            reference_emb=reference_emb,
            lfw_hashes=lfw_hashes,
            global_hashes=global_hashes,
            target_count=args.images_per_id,
            min_res=args.min_resolution,
            sim_threshold=args.sim_threshold,
            hamming_threshold=args.hamming_threshold,
        )

        st = result["status"]
        cnt[st] = cnt.get(st, 0) + 1

        crawl_log[identity] = {
            **result,
            "n_lfw": n_lfw,
            "ref_failed": False,
            "ts": datetime.now().isoformat(),
        }

        pbar.set_postfix_str(
            f"ok={cnt['ok']} parcial={cnt['parcial']} "
            f"falha={cnt['falha']+cnt['no_ref']} skip={cnt['skipped']}",
            refresh=True,
        )

        # Salvar log incrementalmente a cada 25 identidades
        if sum(cnt.values()) % 25 == 0:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(crawl_log, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    pbar.close()
    elapsed = datetime.now() - t0

    # --- Salvar log final ---
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(crawl_log, f, indent=2, ensure_ascii=False)

    # --- Gerar CSV ---
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nome", "id_milvus", "n_lfw_imgs", "n_coletadas", "status"])
        for name, n_lfw in identities:
            d = output_dir / name
            n_col = len(list(d.glob("web_*"))) if d.exists() else 0
            st = crawl_log.get(name, {}).get("status", "?")
            w.writerow([name.replace("_", " "), name, n_lfw, n_col, st])

    # --- Gerar relatorio ---
    report_text = generate_report(identities, output_dir, crawl_log, elapsed, args)
    print(report_text)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # -----------------------------------------------------------------------
    # 7. Watermark check (opcional)
    # -----------------------------------------------------------------------
    if not args.check_watermarks:
        print("\n  Watermark check desativado. Use --check-watermarks para ativar.")
        return

    print("\n" + "=" * 72)
    print("  [Fase 2/2] Verificando marcas d'agua via Ollama VLM...")
    print("=" * 72)

    ollama_client = init_ollama(args.ollama_model, args.ollama_host)

    all_imgs = [
        (name, f)
        for name, _ in identities
        for f in sorted((output_dir / name).glob("web_*"))
        if (output_dir / name).exists()
    ]

    wm_stats = {"clean": 0, "watermark": 0, "error": 0}
    watermarked: List[Dict] = []

    for name, img_path in tqdm(all_imgs, desc="Watermark check", unit="img"):
        has_wm, raw = check_watermark(ollama_client, img_path, args.ollama_model)
        if "ERROR" in raw:
            wm_stats["error"] += 1
        elif has_wm:
            wm_stats["watermark"] += 1
            watermarked.append({"identity": name, "file": img_path.name, "path": str(img_path)})
        else:
            wm_stats["clean"] += 1

    print(
        f"\n  Limpas: {wm_stats['clean']}  "
        f"Com watermark: {wm_stats['watermark']}  "
        f"Erros: {wm_stats['error']}"
    )

    if not watermarked:
        print("  Dataset limpo — sem watermarks detectados.")
        return

    # Substituir imagens com watermark
    print(f"\n  Substituindo {len(watermarked)} imagens com watermark...")
    replaced = 0
    failed = 0

    for item in tqdm(watermarked, desc="Substituindo", unit="img"):
        ip = Path(item["path"])
        td = ip.parent
        name = item["identity"]

        ref = build_reference_embedding(model, lfw_dir, name)
        if ref is None:
            failed += 1
            item["resolved"] = False
            continue

        lfw_h = compute_lfw_hashes(lfw_dir, name)

        ip.unlink(missing_ok=True)

        if recrawl_single(
            name, td, item["file"],
            model, ref, lfw_h,
            global_hashes,
            args.min_resolution,
            args.sim_threshold,
            args.hamming_threshold,
        ):
            replaced += 1
            item["resolved"] = True
        else:
            failed += 1
            item["resolved"] = False

        time.sleep(0.3)

    print(f"\n  Substituidas: {replaced}  Falhas: {failed}")

    wm_log_path = output_dir / "log_watermark.json"
    with open(wm_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": {**wm_stats, "replaced": replaced, "failed": failed},
            "watermarks": watermarked,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"Log de watermark salvo: {wm_log_path}")


if __name__ == "__main__":
    main()
