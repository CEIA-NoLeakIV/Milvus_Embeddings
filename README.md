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
```

## Pesos dos Modelos

Baixe os pesos em: **https://huggingface.co/NoLeak/Embeddings-Models/tree/main**

Coloque na pasta `models/weights/` com os nomes:

```
models/weights/
├── mobilenetv3_large.ckpt
├── mobilenetv3_large_iti.ckpt
└── resnet50_cosface.ckpt
```

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

### Popular o banco com LFW

```bash
python populatemilvus.py
```

Opções:
- `--model cosface_resnet50` - Usar outro modelo
- `--limit 100` - Limitar quantidade de imagens
- `--recreate` - Recriar collection (apaga dados)
- `--no-face-detection` - Desabilitar detecção facial
- `--skip-no-face` - Pular imagens sem face detectada
- `--face-conf 0.35` - Ajustar threshold de confiança

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
├── app/                  # API Flask
│   ├── api.py
│   ├── config.py
│   └── milvus_client.py
├── models/
│   ├── architectures/    # Redes neurais
│   └── weights/          # Pesos (.ckpt)
├── utils/                # Utilitários
│   └── face_detection.py # Detecção e alinhamento facial
├── streamlit_app/        # Interface web
├── tests/                # Testes
├── data/                 # Banco Milvus (auto-gerado)
├── preprocessing.py      # Pré-processamento centralizado
├── populatemilvus.py     # Popular banco
├── run_api.py            # Iniciar API
└── run_streamlit.py      # Iniciar Streamlit
```
