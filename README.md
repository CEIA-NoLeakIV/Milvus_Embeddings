# Face Recognition API

API para extração de embeddings faciais e busca por similaridade usando Milvus.

## Modelos Disponíveis

- **MobileNetV3 Large** (`mobilenetv3_large`, `mobilenetv3_large_iti`)
- **ResNet50 CosFace** (`cosface_resnet50`)
- **TopoFR** — variantes R50/R100/R200 treinadas em MS1MV2 (`topofr_r50_ms1mv2`, `topofr_r100_ms1mv2`, `topofr_r200_ms1mv2`) e em Glint360K (`topofr_r50_glint`, `topofr_r100_glint`, `topofr_r200_glint`)
- **LVFace** — variante B treinada em Glint360K (`lvface_b_glint`)

## Requisitos

- **Sistema Operacional:** Linux (Ubuntu) ou WSL2 no Windows
- **Python:** 3.10+

## Instalação

```bash
# Clonar repositório
git clone <seu-repositorio>
cd Milvus_Embeddings

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Pesos dos Modelos

Baixe os pesos em: **https://huggingface.co/NoLeak/Embeddings-Models/tree/main**

Coloque na pasta `models/weights/` com os nomes exatos abaixo (são os esperados por `Config.MODEL_WEIGHTS`):

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

> Você não precisa baixar todos — qualquer script que itera sobre modelos
> (ex.: `validar_dataset.py`) usa `models_with_weights()` para descobrir
> quais pesos estão presentes em disco e roda apenas esses.

## Detecção Facial

O sistema utiliza **RetinaFace** (via uniface) para detecção, crop e alinhamento facial automático. Isso garante que apenas a região da face seja processada, melhorando a qualidade dos embeddings.

Pipeline de processamento:
1. Detecção de face com RetinaFace
2. Extração de landmarks (5 pontos)
3. Alinhamento facial usando transformação de similaridade
4. Crop para 112x112 pixels
5. Extração do embedding

Configurações padrão:
- **Threshold de confiança:** 0.35
- **Seleção:** Maior face (quando múltiplas detectadas)

## Comandos

### Popular o banco

> **Convenção: 1 collection por modelo/peso.**
> Cada modelo tem **exatamente uma** collection Milvus (nome em
> `Config.MODEL_COLLECTIONS`). Rodar `populatemilvus.py` para um modelo é uma
> operação **destrutiva**: apaga qualquer collection daquele modelo
> (canônica + variantes herdadas com sufixo, ex: `_lfw_ext_val`, `_ext_val`)
> e cria uma nova do zero. Isso elimina o risco de várias collections do
> mesmo modelo coexistindo com dados divergentes.

```bash
python populatemilvus.py
```

Opções:
- `--model cosface_resnet50` — usar outro modelo (default: `mobilenetv3_large`)
- `--lfw-dir <path>` — diretório das imagens (default: `./lfw`)
- `--limit 100` — limitar quantidade de imagens
- `--no-face-detection` — desabilitar detecção facial
- `--skip-no-face` — pular silenciosamente imagens sem face detectada
- `--face-conf 0.35` — ajustar threshold de confiança da detecção

> **Não existe mais `--recreate`** — todo `populate` é destrutivo por design.
> Scripts que consomem a collection (`failure_analysis.py`, `validar_dataset.py`,
> API, Streamlit) derivam o nome **automaticamente** do `--model` via
> `Config.get_collection_name(model)`; nenhum deles aceita override de nome
> de collection.

### Rodar a API

```bash
python run_api.py
```

A API roda em `http://localhost:5000`

### Rodar o Streamlit

```bash
python run_streamlit.py
```

Interface em `http://localhost:8501`

### Rodar testes

```bash
pytest tests/ -v
```

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
  -F "model=mobilenetv3_large" \
  -F "top_k=5"
```

**Inserir no banco:**
```bash
curl -X POST http://localhost:5000/api/milvus/insert \
  -F "images=@foto.jpg" \
  -F "model=mobilenetv3_large" \
  -F "person_id=pessoa_123"
```

**Gerar embedding:**
```bash
curl -X POST http://localhost:5000/api/embedding \
  -F "image=@foto.jpg" \
  -F "model=mobilenetv3_large"
```

**Gerar embedding sem detecção facial:**
```bash
curl -X POST http://localhost:5000/api/embedding \
  -F "image=@foto.jpg" \
  -F "model=mobilenetv3_large" \
  -F "use_face_detection=false"
```

## Estrutura

```
├── app/                                      # API Flask
│   ├── api.py
│   ├── config.py                             # Modelos, pesos, collections, paths
│   └── milvus_client.py                      # Wrapper Milvus (insert/search/purge)
├── data/                                     # Banco Milvus (auto-gerado)
├── face_module/                              # Forks adaptados: TopoFR, LVFace, TransFace
├── models/
│   ├── architectures/                        # Redes neurais (iresnet, mobilenetv3, ...)
│   └── weights/                              # Pesos (.ckpt / .pt)
├── streamlit_app/                            # Interface web
├── tests/                                    # Testes
├── utils/
│   └── face_detection.py                     # SCRFD + alinhamento facial
├── construir_dataset_sanityframework_lfw.py  # Pipeline atual: crawl + 6 critérios sanity
├── failure_analysis.py                       # Análise de falhas + triagem manual + enrichment
├── populatemilvus.py                         # Popular Milvus (1 collection/modelo, destrutivo)
├── preprocessing.py                          # Pré-processamento centralizado
├── requirements.txt                          # Lista com dependências necessárias
├── run_api.py                                # Iniciar API Flask
├── run_streamlit.py                          # Iniciar Streamlit
└── validar_dataset.py                        # Popular + avaliar modelos (leave-one-out, HR@1/5, MRR, AUC)
```
