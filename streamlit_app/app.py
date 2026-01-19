"""
Face Recognition - Streamlit Interface
=======================================
Interface para busca de faces similares no banco Milvus.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import io

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import Config
from app.milvus_client import MilvusClient
from models import ModelFactory


# ===========================================
# Configuração da Página
# ===========================================
st.set_page_config(
    page_title="Face Recognition - Busca por Similaridade",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===========================================
# Cache de recursos
# ===========================================
@st.cache_resource
def load_models():
    """Carrega os modelos disponíveis (cached)."""
    models = {}
    for model_name in Config.AVAILABLE_MODELS:
        try:
            models[model_name] = ModelFactory.create(model_name)
            print(f"✓ Modelo carregado: {model_name}")
        except Exception as e:
            print(f"✗ Erro ao carregar {model_name}: {e}")
    return models


@st.cache_resource
def get_milvus_client():
    """Retorna cliente Milvus (cached)."""
    return MilvusClient()


# ===========================================
# Funções auxiliares
# ===========================================
def extract_embedding(model, image: Image.Image) -> np.ndarray:
    """Extrai embedding de uma imagem PIL."""
    return model.extract_embedding_from_pil(image)


def search_similar_faces(client: MilvusClient, embedding: np.ndarray, top_k: int = 5):
    """Busca faces similares no Milvus."""
    return client.search(embedding, top_k=top_k)


def display_results(results: list):
    """Exibe os resultados da busca."""
    if not results:
        st.warning("Nenhum resultado encontrado.")
        return
    
    st.subheader(f"🎯 Top {len(results)} Resultados")
    
    # Criar colunas para exibir resultados
    cols = st.columns(min(len(results), 5))
    
    for idx, (col, result) in enumerate(zip(cols, results)):
        with col:
            # Card de resultado
            st.markdown(f"**#{idx + 1}**")
            
            # Tentar carregar imagem se existir
            image_path = result.get('image_path', '')
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    st.image(img, use_container_width=True)
                except Exception:
                    st.info("📷 Imagem não disponível")
            else:
                st.info("📷 Imagem não encontrada")
            
            # Informações
            similarity = result.get('distance', 0)
            person_id = result.get('person_id', 'N/A')
            
            # Similaridade como porcentagem (Milvus retorna distância cosseno)
            similarity_pct = similarity * 100
            
            st.metric(
                label="Similaridade",
                value=f"{similarity_pct:.2f}%"
            )
            st.caption(f"**ID:** {person_id}")
            st.caption(f"**Path:** {Path(image_path).name if image_path else 'N/A'}")


def get_collection_stats(client: MilvusClient) -> dict:
    """Obtém estatísticas da collection."""
    try:
        return client.get_collection_stats()
    except Exception:
        return {"row_count": 0}


# ===========================================
# Interface Principal
# ===========================================
def main():
    # Header
    st.title("🔍 Face Recognition")
    st.markdown("**Busca por similaridade facial usando embeddings vetoriais**")
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Seleção de modelo
        selected_model = st.selectbox(
            "Modelo",
            options=Config.AVAILABLE_MODELS,
            index=0,
            help="Escolha o modelo para extração de embeddings"
        )
        
        # Número de resultados
        top_k = st.slider(
            "Número de resultados",
            min_value=1,
            max_value=20,
            value=5,
            help="Quantidade de faces similares a retornar"
        )
        
        st.divider()
        
        # Status do sistema
        st.header("📊 Status")
        
        # Carregar recursos
        with st.spinner("Carregando modelos..."):
            models = load_models()
        
        with st.spinner("Conectando ao Milvus..."):
            milvus_client = get_milvus_client()
        
        # Status dos modelos
        st.markdown("**Modelos:**")
        for model_name in Config.AVAILABLE_MODELS:
            if model_name in models:
                st.success(f"✓ {model_name}")
            else:
                st.error(f"✗ {model_name}")
        
        # Status do Milvus
        st.markdown("**Milvus:**")
        try:
            stats = get_collection_stats(milvus_client)
            row_count = stats.get('row_count', 0)
            st.success(f"✓ Conectado")
            st.info(f"📁 {row_count} embeddings na collection")
        except Exception as e:
            st.error(f"✗ Erro: {e}")
        
        st.divider()
        
        # Informações
        st.header("ℹ️ Informações")
        st.markdown(f"""
        - **Collection:** `{Config.COLLECTION_NAME}`
        - **Dimensão:** {Config.EMBEDDING_DIM}
        - **Métrica:** Similaridade Cosseno
        """)
    
    # Área principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload de Imagem")
        
        uploaded_file = st.file_uploader(
            "Escolha uma imagem",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Faça upload de uma imagem facial para buscar similares"
        )
        
        if uploaded_file is not None:
            # Exibir imagem enviada
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Imagem enviada", use_container_width=True)
            
            # Informações da imagem
            st.caption(f"**Arquivo:** {uploaded_file.name}")
            st.caption(f"**Tamanho:** {image.size[0]}x{image.size[1]}")
            
            # Botão de busca
            search_button = st.button(
                "🔍 Buscar Similares",
                type="primary",
                use_container_width=True
            )
        else:
            search_button = False
            st.info("👆 Faça upload de uma imagem para começar")
    
    with col2:
        st.subheader("📋 Resultados")
        
        if uploaded_file is not None and search_button:
            # Verificar se modelo está carregado
            if selected_model not in models:
                st.error(f"Modelo '{selected_model}' não está disponível.")
                return
            
            model = models[selected_model]
            
            # Extrair embedding
            with st.spinner(f"Extraindo embedding com {selected_model}..."):
                try:
                    embedding = extract_embedding(model, image)
                    st.success(f"✓ Embedding extraído ({len(embedding)} dimensões)")
                except Exception as e:
                    st.error(f"Erro ao extrair embedding: {e}")
                    return
            
            # Buscar similares
            with st.spinner("Buscando faces similares..."):
                try:
                    results = search_similar_faces(milvus_client, embedding, top_k=top_k)
                except Exception as e:
                    st.error(f"Erro na busca: {e}")
                    return
            
            # Exibir resultados
            display_results(results)
            
            # Detalhes em tabela
            if results:
                st.divider()
                st.subheader("📊 Detalhes")
                
                # Preparar dados para tabela
                table_data = []
                for idx, result in enumerate(results):
                    table_data.append({
                        "Rank": idx + 1,
                        "Similaridade": f"{result.get('distance', 0) * 100:.2f}%",
                        "Person ID": result.get('person_id', 'N/A'),
                        "Image Path": result.get('image_path', 'N/A'),
                        "Created At": result.get('created_at', 'N/A')
                    })
                
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True
                )
        
        elif uploaded_file is None:
            # Placeholder
            st.markdown(
                """
                <div style="
                    border: 2px dashed #ccc;
                    border-radius: 10px;
                    padding: 50px;
                    text-align: center;
                    color: #888;
                ">
                    <h3>👈 Faça upload de uma imagem</h3>
                    <p>Os resultados aparecerão aqui</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ===========================================
# Entry Point
# ===========================================
if __name__ == "__main__":
    main()