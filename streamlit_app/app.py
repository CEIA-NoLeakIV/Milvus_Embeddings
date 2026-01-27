import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import io

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import Config
from app.milvus_client import MilvusClient
from models import ModelFactory
from preprocessing import extract_embedding_standardized

try:
    from utils.face_detection import (
        detect_and_align_face,
        detect_faces,
        is_face_detection_available,
        NoFaceDetectedError,
        FaceDetector
    )
    FACE_DETECTION_AVAILABLE = is_face_detection_available()
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    NoFaceDetectedError = Exception


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
            models[model_name] = ModelFactory.create(
                model_name,
                use_tta=Config.USE_TTA
            )
            print(f"✓ Modelo carregado: {model_name} (TTA={Config.USE_TTA})")
        except Exception as e:
            print(f"✗ Erro ao carregar {model_name}: {e}")
    return models


@st.cache_resource
def get_milvus_client(collection_name: str):
    """Retorna cliente Milvus para uma collection específica (cached)."""
    return MilvusClient(collection_name=collection_name)


@st.cache_resource
def get_face_detector():
    if FACE_DETECTION_AVAILABLE:
        return FaceDetector(conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD)
    return None


# ===========================================
# Funções auxiliares
# ===========================================
def extract_embedding(model, uploaded_file, use_face_detection: bool = True) -> np.ndarray:
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    
    return extract_embedding_standardized(
        model,
        file_bytes=file_bytes,
        use_tta=Config.USE_TTA,
        use_face_detection=use_face_detection,
        conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD,
        select_largest=Config.FACE_DETECTION_SELECT_LARGEST
    )


def detect_and_show_face(image: Image.Image) -> tuple:
    if not FACE_DETECTION_AVAILABLE:
        return None, {"error": "Face detection not available"}
    
    try:
        image_np = np.array(image)
        detector = get_face_detector()
        info = detector.get_detection_info(image_np)
        
        if info['num_faces'] == 0:
            return None, {"error": "no_face_detected", "info": info}
        
        aligned_np = detector.detect_and_align(
            image_np,
            select_largest=Config.FACE_DETECTION_SELECT_LARGEST
        )
        aligned_pil = Image.fromarray(aligned_np)
        return aligned_pil, info
        
    except Exception as e:
        return None, {"error": str(e)}


def search_similar_faces(client: MilvusClient, embedding: np.ndarray, top_k: int = 5):
    return client.search(embedding, top_k=top_k)


def display_results(results: list):
    if not results:
        st.warning("Nenhum resultado encontrado.")
        return
    
    st.subheader(f"🎯 Top {len(results)} Resultados")
    cols = st.columns(min(len(results), 5))
    
    for idx, (col, result) in enumerate(zip(cols, results)):
        with col:
            st.markdown(f"**#{idx + 1}**")
            image_path = result.get('image_path', '')
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    st.image(img, use_container_width=True)
                except Exception:
                    st.info("📷 Imagem não disponível")
            else:
                st.info("📷 Imagem não encontrada")
            
            similarity = result.get('distance', 0)
            person_id = result.get('person_id', 'N/A')
            st.metric(label="Similaridade", value=f"{similarity * 100:.2f}%")
            st.caption(f"**ID:** {person_id}")
            st.caption(f"**Path:** {Path(image_path).name if image_path else 'N/A'}")


def get_collection_stats(client: MilvusClient) -> dict:
    try:
        return client.get_collection_stats()
    except Exception:
        return {"row_count": 0}


# ===========================================
# Interface Principal
# ===========================================
def main():
    st.title("🔍 Face Recognition")
    st.markdown("**Busca por similaridade facial usando embeddings vetoriais**")
    
    if FACE_DETECTION_AVAILABLE:
        st.markdown("*✅ Detecção facial ativa (RetinaFace)*")
    else:
        st.markdown("*⚠️ Detecção facial não disponível*")
    
    st.divider()
    
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        selected_model = st.selectbox(
            "Modelo / Collection",
            options=Config.AVAILABLE_MODELS,
            index=0,
            help="Escolha o modelo e a collection correspondente"
        )
        
        # Determina a collection baseada no modelo selecionado
        current_collection = Config.get_collection_name(selected_model)
        st.caption(f"📁 Collection: `{current_collection}`")
        
        top_k = st.slider("Número de resultados", 1, 20, 5)
        st.divider()
        
        st.header("🎯 Detecção Facial")
        if FACE_DETECTION_AVAILABLE:
            use_face_detection = st.checkbox("Habilitar detecção", value=Config.USE_FACE_DETECTION)
            if use_face_detection:
                show_aligned_face = st.checkbox("Mostrar face alinhada", value=True)
                conf_threshold = st.slider("Confiança mínima", 0.1, 1.0, Config.FACE_DETECTION_CONF_THRESHOLD, 0.05)
            else:
                show_aligned_face = False
                conf_threshold = Config.FACE_DETECTION_CONF_THRESHOLD
        else:
            st.warning("⚠️ Instale uniface")
            use_face_detection = False
            show_aligned_face = False
            conf_threshold = 0.5
        
        st.divider()
        st.header("📊 Status")
        
        with st.spinner("Carregando modelos e banco..."):
            models = load_models()
            # Carrega o cliente específico para a collection selecionada
            milvus_client = get_milvus_client(current_collection)
        
        try:
            stats = get_collection_stats(milvus_client)
            row_count = stats.get('row_count', 0)
            st.success(f"✓ Conectado")
            st.info(f"📁 {row_count} embeddings na collection atual")
        except Exception as e:
            st.error(f"✗ Erro: {e}")
            
    # Área principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload de Imagem")
        uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
        
        if uploaded_file is not None:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Imagem original", use_container_width=True)
            
            if use_face_detection and show_aligned_face and FACE_DETECTION_AVAILABLE:
                st.divider()
                st.markdown("**🎯 Face Detectada:**")
                with st.spinner("Detectando face..."):
                    aligned_face, detection_info = detect_and_show_face(image)
                
                if aligned_face is not None:
                    st.image(aligned_face, caption="Face alinhada (112x112)", width=150)
                else:
                    st.error("❌ Nenhuma face detectada ou erro")
            
            st.divider()
            search_button = st.button("🔍 Buscar Similares", type="primary", use_container_width=True)
        else:
            search_button = False
            st.info("👆 Faça upload de uma imagem")
    
    with col2:
        st.subheader("📋 Resultados")
        
        if uploaded_file is not None and search_button:
            if selected_model not in models:
                st.error(f"Modelo indisponível.")
                return
            
            model = models[selected_model]
            
            with st.spinner(f"Extraindo embedding com {selected_model}..."):
                try:
                    uploaded_file.seek(0)
                    embedding = extract_embedding(model, uploaded_file, use_face_detection=use_face_detection)
                    st.success(f"✓ Embedding extraído")
                except NoFaceDetectedError:
                    st.error("❌ Nenhuma face detectada na imagem!")
                    return
                except Exception as e:
                    st.error(f"Erro: {e}")
                    return
            
            with st.spinner(f"Buscando em {current_collection}..."):
                try:
                    results = search_similar_faces(milvus_client, embedding, top_k=top_k)
                except Exception as e:
                    st.error(f"Erro na busca: {e}")
                    return
            
            display_results(results)


if __name__ == "__main__":
    main()