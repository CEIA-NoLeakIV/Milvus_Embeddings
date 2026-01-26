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

# IMPORTANTE: Usar preprocessing centralizado
from preprocessing import extract_embedding_standardized

# Importar módulo de detecção facial para feedback visual
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
            # IMPORTANTE: Usar explicitamente Config.USE_TTA
            models[model_name] = ModelFactory.create(
                model_name,
                use_tta=Config.USE_TTA
            )
            print(f"✓ Modelo carregado: {model_name} (TTA={Config.USE_TTA})")
        except Exception as e:
            print(f"✗ Erro ao carregar {model_name}: {e}")
    return models


@st.cache_resource
def get_milvus_client():
    """Retorna cliente Milvus (cached)."""
    return MilvusClient()


@st.cache_resource
def get_face_detector():
    """Retorna detector facial (cached)."""
    if FACE_DETECTION_AVAILABLE:
        return FaceDetector(
            conf_threshold=Config.FACE_DETECTION_CONF_THRESHOLD
        )
    return None


# ===========================================
# Funções auxiliares
# ===========================================
def extract_embedding(
    model,
    uploaded_file,
    use_face_detection: bool = True
) -> np.ndarray:
    """
    Extrai embedding de um arquivo uploaded do Streamlit.
    USA PREPROCESSING CENTRALIZADO.
    
    Args:
        model: Modelo de face recognition
        uploaded_file: Arquivo do st.file_uploader
        use_face_detection: Se True, aplica detecção facial
    
    Returns:
        Embedding numpy array
    """
    # IMPORTANTE: Ler bytes e usar preprocessing centralizado
    file_bytes = uploaded_file.read()
    
    # Resetar o ponteiro do arquivo caso precise ser usado novamente
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
    """
    Detecta face na imagem e retorna informações para exibição.
    
    Args:
        image: Imagem PIL
    
    Returns:
        Tuple (aligned_face_pil, detection_info) ou (None, error_message)
    """
    if not FACE_DETECTION_AVAILABLE:
        return None, {"error": "Face detection not available"}
    
    try:
        # Converter para numpy
        image_np = np.array(image)
        
        # Detectar faces
        detector = get_face_detector()
        info = detector.get_detection_info(image_np)
        
        if info['num_faces'] == 0:
            return None, {"error": "no_face_detected", "info": info}
        
        # Alinhar face
        aligned_np = detector.detect_and_align(
            image_np,
            select_largest=Config.FACE_DETECTION_SELECT_LARGEST
        )
        
        # Converter para PIL
        aligned_pil = Image.fromarray(aligned_np)
        
        return aligned_pil, info
        
    except Exception as e:
        return None, {"error": str(e)}


def search_similar_faces(client: MilvusClient, embedding: np.ndarray, top_k: int = 5):
    """Busca faces similares no Milvus."""
    return client.search(embedding, top_k=top_k)


def display_results(results: list):
    """Exibe os resultados da busca."""
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
    st.title("🔍 Face Recognition")
    st.markdown("**Busca por similaridade facial usando embeddings vetoriais**")
    
    # Mostrar status da detecção facial
    if FACE_DETECTION_AVAILABLE:
        st.markdown("*✅ Detecção facial ativa (RetinaFace)*")
    else:
        st.markdown("*⚠️ Detecção facial não disponível*")
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        selected_model = st.selectbox(
            "Modelo",
            options=Config.AVAILABLE_MODELS,
            index=0,
            help="Escolha o modelo para extração de embeddings"
        )
        
        top_k = st.slider(
            "Número de resultados",
            min_value=1,
            max_value=20,
            value=5,
            help="Quantidade de faces similares a retornar"
        )
        
        st.divider()
        
        # Configurações de detecção facial
        st.header("🎯 Detecção Facial")
        
        if FACE_DETECTION_AVAILABLE:
            use_face_detection = st.checkbox(
                "Habilitar detecção",
                value=Config.USE_FACE_DETECTION,
                help="Detecta, recorta e alinha a face automaticamente"
            )
            
            if use_face_detection:
                show_aligned_face = st.checkbox(
                    "Mostrar face alinhada",
                    value=True,
                    help="Exibe a face após detecção e alinhamento"
                )
                
                conf_threshold = st.slider(
                    "Confiança mínima",
                    min_value=0.1,
                    max_value=1.0,
                    value=Config.FACE_DETECTION_CONF_THRESHOLD,
                    step=0.05,
                    help="Limiar de confiança do detector"
                )
            else:
                show_aligned_face = False
                conf_threshold = Config.FACE_DETECTION_CONF_THRESHOLD
        else:
            st.warning("⚠️ Instale uniface para habilitar")
            st.code("pip install uniface", language="bash")
            use_face_detection = False
            show_aligned_face = False
            conf_threshold = 0.5
        
        st.divider()
        
        st.header("📊 Status")
        
        with st.spinner("Carregando modelos..."):
            models = load_models()
        
        with st.spinner("Conectando ao Milvus..."):
            milvus_client = get_milvus_client()
        
        st.markdown("**Modelos:**")
        for model_name in Config.AVAILABLE_MODELS:
            if model_name in models:
                st.success(f"✓ {model_name}")
            else:
                st.error(f"✗ {model_name}")
        
        st.markdown("**Milvus:**")
        try:
            stats = get_collection_stats(milvus_client)
            row_count = stats.get('row_count', 0)
            st.success(f"✓ Conectado")
            st.info(f"📁 {row_count} embeddings")
        except Exception as e:
            st.error(f"✗ Erro: {e}")
        
        st.divider()
        
        st.header("ℹ️ Informações")
        st.markdown(f"""
        - **Collection:** `{Config.COLLECTION_NAME}`
        - **Dimensão:** {Config.EMBEDDING_DIM}
        - **TTA:** {'ON' if Config.USE_TTA else 'OFF'}
        - **Face Detection:** {'ON' if use_face_detection else 'OFF'}
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
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert('RGB')
            
            st.image(image, caption="Imagem original", use_container_width=True)
            st.caption(f"**Arquivo:** {uploaded_file.name}")
            st.caption(f"**Tamanho:** {image.size[0]}x{image.size[1]}")
            
            # Mostrar face detectada/alinhada (se habilitado)
            if use_face_detection and show_aligned_face and FACE_DETECTION_AVAILABLE:
                st.divider()
                st.markdown("**🎯 Face Detectada:**")
                
                with st.spinner("Detectando face..."):
                    aligned_face, detection_info = detect_and_show_face(image)
                
                if aligned_face is not None:
                    # Exibir face alinhada
                    st.image(aligned_face, caption="Face alinhada (112x112)", width=150)
                    
                    # Informações da detecção
                    if 'faces' in detection_info and len(detection_info['faces']) > 0:
                        best_face = detection_info['faces'][0]
                        st.caption(f"**Score:** {best_face['score']:.2f}")
                        st.caption(f"**Faces encontradas:** {detection_info['num_faces']}")
                        
                        if detection_info['num_faces'] > 1:
                            st.info(f"ℹ️ {detection_info['num_faces']} faces detectadas. Usando a maior.")
                else:
                    # Erro na detecção
                    error = detection_info.get('error', 'Erro desconhecido')
                    if error == 'no_face_detected':
                        st.error("❌ Nenhuma face detectada na imagem")
                        st.caption("Tente com outra imagem ou desabilite a detecção facial.")
                    else:
                        st.error(f"❌ Erro: {error}")
            
            st.divider()
            
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
            if selected_model not in models:
                st.error(f"Modelo '{selected_model}' não está disponível.")
                return
            
            model = models[selected_model]
            
            # Extrair embedding usando preprocessing centralizado
            with st.spinner(f"Extraindo embedding com {selected_model}..."):
                try:
                    # Resetar ponteiro do arquivo
                    uploaded_file.seek(0)
                    embedding = extract_embedding(
                        model,
                        uploaded_file,
                        use_face_detection=use_face_detection
                    )
                    st.success(f"✓ Embedding extraído ({len(embedding)} dimensões)")
                    
                except NoFaceDetectedError as e:
                    st.error("❌ Nenhuma face detectada na imagem!")
                    st.warning("""
                    **Possíveis soluções:**
                    - Tente com outra imagem que contenha uma face clara
                    - Verifique se a face está visível e bem iluminada
                    - Desabilite a detecção facial na sidebar (não recomendado)
                    """)
                    return
                    
                except Exception as e:
                    st.error(f"Erro ao extrair embedding: {e}")
                    import traceback
                    st.code(traceback.format_exc())
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
                
                # Info sobre configurações usadas
                st.divider()
                st.caption(f"""
                **Configurações usadas:**
                Modelo: {selected_model} | 
                TTA: {'ON' if Config.USE_TTA else 'OFF'} | 
                Face Detection: {'ON' if use_face_detection else 'OFF'} |
                Dimensão: {len(embedding)}
                """)
        
        elif uploaded_file is None:
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
                    <br>
                    <p style="font-size: 12px; color: #aaa;">
                        ✓ Pré-processamento padronizado ativo<br>
                        """ + ("✓ Detecção facial ativa" if FACE_DETECTION_AVAILABLE else "⚠️ Detecção facial não disponível") + """
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()