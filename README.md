# Face Recognition API

API para extração de embeddings faciais e busca por similaridade usando Milvus.

## Modelos Disponíveis

- **MobileNetV3 Large**
- **ResNet50 CosFace**

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
pip install "pymilvus[milvus_lite]"
```

## Pesos dos Modelos

Baixe os pesos em: **https://huggingface.co/NoLeak/Embeddings-Models/tree/main**

Coloque na pasta `models/weights/` com os nomes:

```
models/weights/
├── mobilenetv3_large.ckpt
└── resnet50_cosface.ckpt
```

## Comandos

### Popular o banco com LFW

```bash
python populatemilvus.py
```

Opções:
- `--model cosface_resnet50` - Usar outro modelo
- `--limit 100` - Limitar quantidade de imagens
- `--recreate` - Recriar collection (apaga dados)

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
cd tests
python test_api.py
```

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/health` | Status da API |
| GET | `/api/models` | Listar modelos |
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

## Estrutura

```
├── app/                  # API Flask
├── models/
│   ├── architectures/    # Redes neurais
│   └── weights/          # Pesos (.ckpt)
├── streamlit_app/        # Interface web
├── tests/                # Testes
├── data/                 # Banco Milvus (auto-gerado)
├── populatemilvus.py     # Popular banco
├── preprocessing.py      # Centraliza funções de pré-processamento
├── run_api.py            # Iniciar API
└── run_streamlit.py      # Iniciar Streamlit
```
