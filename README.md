# Face Recognition API

API para extração de embeddings faciais e busca por similaridade usando Milvus.

## Modelos Disponíveis

- **MobileNetV3 Large** — `mobilenetv3_large` / `mobilenetv3_large_iti`
- **ResNet50 CosFace** — `cosface_resnet50`
- **TopoFR** — `topofr_r{50,100,200}_{ms1mv2,glint}`
- **LVFace ViT-B** — `lvface_b_glint`

Todos os modelos são gerenciados pelo `ModelFactory` com lazy loading e cache automático.

## Requisitos

- **Sistema Operacional:** Linux (Ubuntu) ou WSL2 no Windows
- **Python:** 3.10+

## Instalação

```bash
git clone <seu-repositorio>
cd Milvus_Embeddings

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Pesos dos Modelos

Baixe os pesos em: **https://huggingface.co/NoLeak/Embeddings-Models/tree/main**

Coloque na pasta `models/weights/`:

```
models/weights/
├── mobilenetv3_large.ckpt
├── mobilenetv3_large_iti.ckpt
├── resnet50_cosface.ckpt
├── MS1MV2_R50_TopoFR_9649.pt
├── MS1MV2_R100_TopoFR_9695.pt
├── MS1MV2_R200_TopoFR_9708.pt
├── Glint360K_R50_TopoFR_9727.pt
├── Glint360K_R100_TopoFR_9760.pt
├── Glint360K_R200_TopoFR_9784.pt
└── LVFace-B_Glint360K.pt
```

## Pipeline de Processamento

```
Imagem → SCRFD (detecção) → norm_crop (alinhamento 112×112) → Embedding (512-d) → TTA (1024-d) → L2 Norm → Milvus
```

**Detecção facial:** SCRFD (InsightFace) com `face_align.norm_crop` para alinhamento. Configurações padrão: threshold de confiança 0.35, seleção da maior face quando múltiplas são detectadas.

**TTA (Test-Time Augmentation):** Concatenação do embedding original (512-d) com o embedding da imagem espelhada horizontalmente, gerando vetores de 1024 dimensões.

**Pré-processamento:** Centralizado em `preprocessing.py` via `extract_embedding_standardized()`, garantindo consistência entre API, Streamlit, scripts e testes. Aceita imagens de qualquer fonte (path, bytes, stream, PIL).

## Comandos

### Popular o banco

```bash
python populatemilvus.py
```

Opções: `--model cosface_resnet50`, `--limit 100`, `--recreate`, `--no-face-detection`, `--skip-no-face`, `--face-conf 0.35`

Cada modelo popula sua própria collection (`face_embeddings_<modelo>`).

### Rodar a API

```bash
python run_api.py
```

A API roda em `http://localhost:5000`. Opções: `--host`, `--port`, `--debug`

### Rodar o Streamlit

```bash
python run_streamlit.py
```

Interface em `http://localhost:8501`

### Rodar testes

```bash
pytest tests/ -v
```

### Failure Analysis

Diagnóstico por imagem usando leave-one-out no Milvus:

```bash
python failure_analysis.py --model topofr_r100_glint
python failure_analysis.py --model topofr_r100_glint --visual-report
python failure_analysis.py --model topofr_r100_glint --top-n 50 --only-failures
```

Gera: `per_image_results.csv`, `per_identity_results.csv`, `confusion_pairs.csv`, `summary.json` e, opcionalmente, `visual_report.html` com cards lado a lado (query vs. erro).

### Construir Dataset de Validação Externa

Coleta imagens via Bing com as identidades do LFW, validando cada imagem com TopoFR antes de aceitar:

```bash
python construir_dataset_sanityframework_lfw.py --lfw-dir lfw
python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --images-per-id 10
python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --sim-threshold 0.25
python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --start-from 500
python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --dry-run
python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --check-watermarks
python construir_dataset_sanityframework_lfw.py --lfw-dir lfw --check-face-count
```

**Sanity Framework:** cada imagem baixada passa por 6 critérios antes de ser aceita — integridade, resolução mínima (112px), deduplicação global por hash perceptual, anti-leak LFW (distância Hamming), exatamente 1 face detectada pelo SCRFD, e similaridade coseno TopoFR ≥ 0.30 contra o embedding de referência LFW da identidade. Saída em `dataset_ext_val_lfw/`.

### Validar Dataset

Popula o Milvus com o `dataset_ext_val_lfw` e avalia todos os modelos com disponibilidade de pesos:

```bash
python validar_dataset.py
python validar_dataset.py --models topofr_r100_glint lvface_b_glint
python validar_dataset.py --recreate
python validar_dataset.py --skip-populate
python validar_dataset.py --complete 3
```

Avaliação **leave-one-out** em dois cenários: identidades completas (≥ 5 imagens) e todas as identidades (≥ 2 imagens). Gera `avaliacao_modelos.md` com tabela comparativa (HR@1, HR@5, MRR, AUC, Sim Genuína, Gap) e JSONs individuais por modelo em `resultados/`.

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/health` | Status da API |
| GET | `/api/models` | Listar modelos |
| GET | `/api/face-detection/status` | Status da detecção facial |
| POST | `/api/embedding` | Gerar embedding de 1 imagem |
| POST | `/api/embeddings/batch` | Gerar embeddings em lote |
| POST | `/api/milvus/insert` | Inserir no banco |
| POST | `/api/milvus/search` | Buscar similares |
| GET | `/api/milvus/stats` | Estatísticas do banco |

## Exemplos de Uso

**Buscar similares:**
```bash
curl -X POST http://localhost:5000/api/milvus/search \
  -F "image=@foto.jpg" \
  -F "model=topofr_r100_glint" \
  -F "top_k=5"
```

**Inserir no banco:**
```bash
curl -X POST http://localhost:5000/api/milvus/insert \
  -F "images=@foto.jpg" \
  -F "model=topofr_r100_glint" \
  -F "person_id=pessoa_123"
```

**Gerar embedding:**
```bash
curl -X POST http://localhost:5000/api/embedding \
  -F "image=@foto.jpg" \
  -F "model=topofr_r100_glint"
```

**Gerar embedding sem detecção facial:**
```bash
curl -X POST http://localhost:5000/api/embedding \
  -F "image=@foto.jpg" \
  -F "model=topofr_r100_glint" \
  -F "use_face_detection=false"
```

## Estrutura

```
├── app/                                     # API Flask
│   ├── api.py                               # Rotas e endpoints
│   ├── config.py                            # Configurações centralizadas
│   └── milvus_client.py                     # Cliente Milvus com normalização L2
├── models/
│   ├── base.py                              # BaseModel (classe abstrata)
│   ├── mobilenet_model.py                   # MobileNetV3
│   ├── cosface_model.py                     # CosFace ResNet-50
│   ├── topofr_model.py                      # TopoFR (R50/R100/R200)
│   ├── lvface_model.py                      # LVFace ViT-B
│   ├── __init__.py                          # ModelFactory
│   └── weights/                             # Pesos dos modelos
├── face_module/
│   ├── TopoFR/                              # Código-fonte TopoFR
│   ├── TransFace/                           # Código-fonte TransFace
│   └── LVFace/                              # Código-fonte LVFace (ViT)
├── utils/
│   └── face_detection.py                    # SCRFD + norm_crop
├── streamlit_app/                           # Interface web
├── tests/                                   # Testes (pytest)
├── data/                                    # Banco Milvus (auto-gerado)
├── preprocessing.py                         # Pré-processamento padronizado + TTA
├── populatemilvus.py                        # Popular banco por modelo
├── failure_analysis.py                      # Diagnóstico leave-one-out
├── construir_dataset_sanityframework_lfw.py # Coletar dataset de validação externa
├── validar_dataset.py                       # Avaliar modelos no dataset externo
├── run_api.py                               # Iniciar API
└── run_streamlit.py                         # Iniciar Streamlit
```
