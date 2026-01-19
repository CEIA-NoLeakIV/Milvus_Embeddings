# Face Recognition API

Sistema de reconhecimento facial com API Flask, suporte a múltiplos modelos (MobileNetV3 Large e ResNet50 CosFace) e banco de dados vetorial Milvus Lite.

---

## 🏗️ Arquitetura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   Flask API     │────▶│  Milvus Lite    │
│   Interface     │     │                 │     │  (Local DB)     │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼─────┐           ┌───────▼───────┐
              │ MobileNet │           │ ResNet50      │
              │ V3 Large  │           │ CosFace       │
              │ (512-dim) │           │ (512-dim)     │
              └───────────┘           └───────────────┘
```

---

## 📋 Funcionalidades

- ✅ Extração de embeddings faciais (512 dimensões)
- ✅ Suporte a dois modelos: MobileNetV3 Large e ResNet50 CosFace
- ✅ API REST com endpoints para:
  - Gerar embedding de uma única imagem
  - Gerar embeddings em lote
  - Inserir embeddings no Milvus
- ✅ Interface Streamlit para busca por similaridade
- ✅ Banco de dados vetorial local (Milvus Lite)

---

## 🚀 Instalação

### 1. Clonar o repositório

```bash
git clone <seu-repositorio>
cd face-recognition-api
```

### 2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Adicionar os pesos dos modelos

Coloque os arquivos de checkpoint na pasta `models/weights/`:

```
models/weights/
├── mobilenetv3_large.ckpt    # Peso do MobileNetV3 Large
└── resnet50_cosface.ckpt     # Peso do ResNet50 CosFace
```

### 5. Adicionar as arquiteturas dos modelos

Copie os arquivos de arquitetura para `models/architectures/`:

```
models/architectures/
├── __init__.py
├── mobilenetv3.py    # Do repositório face-recognition
├── resnet.py         # Do repositório cosface
└── layers.py         # utils/layers.py do repositório face-recognition
```

---

## 🖥️ Uso

### Iniciar a API

```bash
python run_api.py
```

Opções:
```bash
python run_api.py --host 0.0.0.0 --port 5000 --debug
```

### Iniciar a Interface Streamlit

```bash
python run_streamlit.py
```

Opções:
```bash
python run_streamlit.py --port 8502
```

---

## 📡 Endpoints da API

### Health Check

```http
GET /api/health
```

**Resposta:**
```json
{
    "status": "healthy",
    "models_loaded": ["mobilenetv3_large", "cosface_resnet50"],
    "milvus_connected": true
}
```

### Listar Modelos Disponíveis

```http
GET /api/models
```

**Resposta:**
```json
{
    "models": [
        {
            "name": "mobilenetv3_large",
            "embedding_dim": 512,
            "description": "MobileNetV3 Large otimizado para reconhecimento facial"
        },
        {
            "name": "cosface_resnet50",
            "embedding_dim": 512,
            "description": "ResNet50 com CosFace Loss"
        }
    ]
}
```

### Gerar Embedding (Única Imagem)

```http
POST /api/embedding
Content-Type: multipart/form-data
```

**Parâmetros:**
| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `image` | file | Sim | Arquivo de imagem |
| `model` | string | Não | Nome do modelo (default: `mobilenetv3_large`) |

**Exemplo com cURL:**
```bash
curl -X POST http://localhost:5000/api/embedding \
  -F "image=@foto.jpg" \
  -F "model=mobilenetv3_large"
```

**Resposta:**
```json
{
    "success": true,
    "model": "mobilenetv3_large",
    "embedding": [0.0123, -0.0456, ...],
    "embedding_dim": 512
}
```

### Gerar Embeddings em Lote

```http
POST /api/embeddings/batch
Content-Type: multipart/form-data
```

**Parâmetros:**
| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `images` | files | Sim | Múltiplos arquivos de imagem |
| `model` | string | Não | Nome do modelo (default: `mobilenetv3_large`) |

**Exemplo com cURL:**
```bash
curl -X POST http://localhost:5000/api/embeddings/batch \
  -F "images=@foto1.jpg" \
  -F "images=@foto2.jpg" \
  -F "images=@foto3.jpg" \
  -F "model=cosface_resnet50"
```

**Resposta:**
```json
{
    "success": true,
    "model": "cosface_resnet50",
    "results": [
        {
            "filename": "foto1.jpg",
            "embedding": [0.0123, ...],
            "success": true
        },
        {
            "filename": "foto2.jpg",
            "embedding": [0.0456, ...],
            "success": true
        }
    ],
    "total": 3,
    "successful": 3,
    "failed": 0
}
```

### Inserir Embeddings no Milvus

```http
POST /api/milvus/insert
Content-Type: multipart/form-data
```

**Parâmetros:**
| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `images` | files | Sim | Arquivos de imagem |
| `model` | string | Não | Nome do modelo |
| `person_id` | string | Sim | ID da pessoa (CPF, nome, etc.) |
| `image_paths` | string | Não | Caminhos originais (JSON array) |

**Exemplo com cURL:**
```bash
curl -X POST http://localhost:5000/api/milvus/insert \
  -F "images=@foto1.jpg" \
  -F "images=@foto2.jpg" \
  -F "model=mobilenetv3_large" \
  -F "person_id=12345678900" \
  -F 'image_paths=["/path/to/foto1.jpg", "/path/to/foto2.jpg"]'
```

**Resposta:**
```json
{
    "success": true,
    "message": "2 embeddings inseridos com sucesso",
    "inserted_count": 2,
    "collection": "face_embeddings"
}
```

---

## 🎨 Interface Streamlit

A interface permite:

1. **Upload de imagem** - Envie uma foto para busca
2. **Seleção de modelo** - Escolha entre MobileNetV3 ou CosFace
3. **Busca por similaridade** - Retorna os 5 rostos mais similares
4. **Visualização dos resultados** - Mostra as imagens e scores de similaridade

---

## 📁 Estrutura do Projeto

```
face-recognition-api/
│
├── app/
│   ├── __init__.py
│   ├── api.py                 # API Flask
│   ├── config.py              # Configurações
│   └── milvus_client.py       # Cliente Milvus
│
├── models/
│   ├── __init__.py
│   ├── base.py                # Classe base
│   ├── mobilenet_model.py     # Wrapper MobileNetV3
│   ├── cosface_model.py       # Wrapper ResNet50
│   │
│   ├── architectures/         # Arquiteturas (copiar dos repos)
│   │   ├── __init__.py
│   │   ├── mobilenetv3.py
│   │   ├── resnet.py
│   │   └── layers.py
│   │
│   └── weights/               # Checkpoints (você adiciona)
│       ├── mobilenetv3_large.ckpt
│       └── resnet50_cosface.ckpt
│
├── streamlit_app/
│   └── app.py                 # Interface Streamlit
│
├── data/
│   └── milvus_face.db         # Banco Milvus (auto-gerado)
│
├── requirements.txt
├── run_api.py
├── run_streamlit.py
└── README.md
```

---

## 🗄️ Schema do Milvus

```python
{
    "id": int,           # Auto-gerado
    "embedding": float[], # 512 dimensões
    "person_id": str,     # CPF ou identificador
    "image_path": str,    # Caminho da imagem
    "created_at": str     # Timestamp auto-gerado
}
```

---

## ⚙️ Configurações

Edite `app/config.py` para personalizar:

```python
class Config:
    # Milvus
    MILVUS_DB_PATH = "./data/milvus_face.db"
    COLLECTION_NAME = "face_embeddings"
    EMBEDDING_DIM = 512
    
    # Modelos
    DEFAULT_MODEL = "mobilenetv3_large"
    WEIGHTS_DIR = "./models/weights"
    
    # API
    MAX_BATCH_SIZE = 100
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
```

---

## 🧪 Testando a API

### Com Python requests

```python
import requests

# Gerar embedding
with open("foto.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/api/embedding",
        files={"image": f},
        data={"model": "mobilenetv3_large"}
    )
    print(response.json())
```

### Com HTTPie

```bash
# Health check
http GET localhost:5000/api/health

# Gerar embedding
http -f POST localhost:5000/api/embedding image@foto.jpg model=mobilenetv3_large
```

---

## 📝 Notas Importantes

1. **GPU**: Se disponível, os modelos utilizarão CUDA automaticamente
2. **Milvus Lite**: O banco é local e persiste em `data/milvus_face.db`
3. **Dimensão**: Ambos os modelos geram embeddings de 512 dimensões
4. **Formato de imagem**: Suporta JPG, PNG, BMP, TIFF

---

## 🐛 Troubleshooting

### Erro: "Checkpoint não encontrado"
Verifique se os arquivos `.ckpt` estão em `models/weights/`

### Erro: "CUDA out of memory"
Reduza o tamanho do lote ou use CPU:
```python
# Em config.py
DEVICE = "cpu"
```

### Erro: "Module not found"
Verifique se as arquiteturas estão em `models/architectures/`

---

## 📄 Licença

MIT License