"""
SISTEMA STREAMLIT PROFESIONAL PARA DETECCIÓN DE COVID-19
Vision Transformer - Versión optimizada para Render (Free Plan)
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import sys
from datetime import datetime
import warnings
import gc
from pathlib import Path
warnings.filterwarnings('ignore')

# ==================== OPTIMIZACIÓN EXTREMA PARA RENDER ====================
# Configurar para uso MÍNIMO de memoria
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configurar PyTorch para uso mínimo de recursos
torch.set_num_threads(1)
if hasattr(torch, 'set_num_interop_threads'):
    torch.set_num_interop_threads(1)

# Usar precisión mixta para ahorrar memoria
torch.set_float32_matmul_precision('medium')

# Configurar matplotlib para usar menos memoria
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 80

# ==================== CONFIGURACIÓN ====================
class Config:
    MODEL_PATH = Path("best_model_gpu.pth")
    CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    CLASS_DESCRIPTIONS = {
        'COVID': "Infección por SARS-CoV-2 detectada en radiografía pulmonar",
        'Lung_Opacity': "Opacidades pulmonares (pueden indicar neumonía, edema, etc.)",
        'Normal': "Radiografía pulmonar sin anomalías detectables",
        'Viral Pneumonia': "Neumonía viral (no COVID-19) detectada"
    }
    CLASS_RECOMMENDATIONS = {
        'COVID': "Consulta médica inmediata. Aislamiento recomendado. Prueba PCR confirmatoria.",
        'Lung_Opacity': "Evaluación médica necesaria. Puede requerir tomografía computarizada.",
        'Normal': "Sin hallazgos patológicos. Continuar con controles rutinarios.",
        'Viral Pneumonia': "Tratamiento antiviral posiblemente requerido. Consulta médica."
    }

# ==================== MODELO ====================
@st.cache_resource(show_spinner=False)
def load_model():
    """Cargar el modelo Vision Transformer - Optimizado para memoria limitada"""
    try:
        # Limpieza agresiva de memoria
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Verificar si el archivo existe
        model_path = Config.MODEL_PATH
        if not model_path.exists():
            st.error(f"Archivo de modelo no encontrado en: {model_path.absolute()}")
            return None, {}, 0
        
        # Detectar si estamos en Render
        is_render = os.environ.get('RENDER') == 'true' or os.environ.get('ON_RENDER')
        
        # Siempre usar CPU en Render para ahorrar memoria
        device = torch.device('cpu')
        map_location = 'cpu'
        
        # Información de carga
        with st.spinner("Cargando modelo (esto puede tomar 1-2 minutos)..."):
            # Cargar checkpoint con optimizaciones
            checkpoint = torch.load(
                str(model_path), 
                map_location=map_location,
                weights_only=False,
                pickle_module=__import__('pickle')  # Usar pickle estándar
            )
            
            # Crear modelo con configuraciones livianas
            model = timm.create_model(
                'vit_base_patch16_224', 
                pretrained=False, 
                num_classes=len(Config.CLASS_NAMES),
                act_layer=nn.ReLU,  # ReLU es más liviano que GELU
                drop_rate=0.0,
                drop_path_rate=0.0
            )
            
            # Cargar pesos
            state_dict = checkpoint['model_state_dict']
            
            # Manejar DataParallel si fue usado
            if any(key.startswith('module.') for key in state_dict.keys()):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict)
            
            # Optimizaciones extremas para Render
            if is_render:
                # Usar float16 para ahorrar 50% de memoria
                model = model.half()
            
            model = model.to(device)
            model.eval()
            
            # Desactivar gradientes para ahorrar memoria
            for param in model.parameters():
                param.requires_grad = False
            
            torch.set_grad_enabled(False)
            
            # Cargar historial de entrenamiento
            history = checkpoint.get('history', {})
            val_acc = checkpoint.get('val_acc', 93.27)
            
            # Liberar memoria del checkpoint
            del checkpoint
            del state_dict
            gc.collect()
        
        return model, history, val_acc
        
    except torch.cuda.OutOfMemoryError:
        st.error("ERROR: Memoria insuficiente. El modelo es muy grande para el plan gratuito.")
        return None, {}, 0
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)[:100]}")
        return None, {}, 0

# ==================== TRANSFORMACIONES ====================
def get_transforms():
    """Obtener transformaciones para las imágenes"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ==================== PREDICCIÓN ====================
def predict_image(model, image):
    """Realizar predicción en una imagen optimizada"""
    try:
        transform = get_transforms()
        
        # Preprocesar imagen
        img_tensor = transform(image).unsqueeze(0)
        
        # Mover al mismo dispositivo que el modelo
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Si el modelo está en half, convertir la imagen también
        if next(model.parameters()).dtype == torch.float16:
            img_tensor = img_tensor.half()
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            predicted_class = Config.CLASS_NAMES[predicted_idx]
            confidence = probabilities[predicted_idx].item()
            
            # Obtener todas las probabilidades
            all_probs = {Config.CLASS_NAMES[i]: prob.item() 
                        for i, prob in enumerate(probabilities)}
        
        # Limpiar memoria
        del img_tensor
        del outputs
        del probabilities
        gc.collect()
        
        return predicted_class, confidence, all_probs
        
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return "Error", 0.0, {}

# ==================== GRÁFICOS CON MATPLOTLIB ====================
def create_training_history_plot(history):
    """Crear gráfico del historial de entrenamiento"""
    if not history or 'train_acc' not in history:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        epochs = list(range(1, len(history['train_acc']) + 1))
        
        ax.plot(epochs, history['train_acc'], 'b-', linewidth=1.5, label='Entrenamiento')
        
        if 'val_acc' in history and history['val_acc']:
            ax.plot(epochs[:len(history['val_acc'])], history['val_acc'], 'r-', 
                    linewidth=1.5, label='Validación')
        
        ax.set_xlabel('Época')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Historial de Entrenamiento - Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig
    except:
        return None

def create_probability_chart(probabilities):
    """Crear gráfico de barras para probabilidades"""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        sorted_indices = np.argsort(probs)[::-1]
        classes = [classes[i] for i in sorted_indices]
        probs = [probs[i] for i in sorted_indices]
        
        bars = ax.bar(classes, probs, color=Config.CLASS_COLORS[:len(classes)])
        
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Clase')
        ax.set_ylabel('Probabilidad')
        ax.set_title('Probabilidades de Predicción')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        return fig
    except:
        return None

# ==================== COMPONENTES UI ====================
def create_header():
    """Crear encabezado de la aplicación"""
    st.title("Sistema de Detección de COVID-19")
    st.markdown("Vision Transformer para análisis de radiografías pulmonares")
    st.markdown("---")

def create_sidebar():
    """Crear barra lateral"""
    st.sidebar.title("Navegación")
    
    page = st.sidebar.radio(
        "Seleccione una página:",
        ["Inicio", "Predicción", "Análisis", "Información"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Información del sistema
    st.sidebar.subheader("Información del Sistema")
    
    if torch.cuda.is_available():
        device_info = "GPU disponible"
    else:
        device_info = "CPU solamente"
    
    st.sidebar.text(f"PyTorch: {torch.__version__}")
    st.sidebar.text(f"Dispositivo: {device_info}")
    
    # Limpieza de memoria manual
    if st.sidebar.button("Limpiar memoria"):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.sidebar.success("Memoria limpiada")
    
    st.sidebar.markdown("---")
    st.sidebar.warning("Sistema de apoyo diagnóstico. No sustituye evaluación médica.")
    
    return page

def create_footer():
    """Crear pie de página"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 10px;'>
        <p>Sistema de Detección COVID-19 | Uso exclusivo para investigación</p>
        <p><small>Herramienta de apoyo diagnóstico. No sustituye evaluación médica profesional.</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== PÁGINAS ====================
def home_page(model, history, val_acc):
    """Página de inicio"""
    st.header("Bienvenido al Sistema de Detección COVID-19")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Características del Sistema:**
        - 4 Clases: COVID, Opacidad Pulmonar, Normal, Neumonía Viral
        - Modelo: Vision Transformer (ViT-Base-224)
        - Precisión: 93.27% en validación
        - Solo para investigación
        """)
    
    with col2:
        st.success("""
        **Cómo Usar:**
        1. Navega a Predicción
        2. Sube una radiografía pulmonar
        3. Obtén el diagnóstico asistido
        4. Consulta siempre con un médico
        """)
    
    # Estado del modelo
    if model is not None:
        st.success("Modelo cargado correctamente")
    else:
        st.error("Modelo no disponible - Problemas de memoria")
    
    st.markdown("---")
    
    # Métricas del modelo
    st.subheader("Rendimiento del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{val_acc:.1f}%")
    
    with col2:
        st.metric("Clases", "4")
    
    with col3:
        st.metric("Épocas", "5")
    
    with col4:
        if torch.cuda.is_available():
            st.metric("Dispositivo", "GPU")
        else:
            st.metric("Dispositivo", "CPU")
    
    st.markdown("---")
    
    # Clases detectables
    st.subheader("Clases Detectables")
    
    cols = st.columns(len(Config.CLASS_NAMES))
    
    for idx, (class_name, color) in enumerate(zip(Config.CLASS_NAMES, Config.CLASS_COLORS)):
        with cols[idx]:
            st.markdown(
                f"""
                <div style='
                    background-color: {color}20;
                    border-left: 4px solid {color};
                    padding: 10px;
                    border-radius: 4px;
                    margin: 2px 0;
                    font-size: 0.9em;
                '>
                <strong>{class_name}</strong><br>
                {Config.CLASS_DESCRIPTIONS[class_name][:50]}...
                </div>
                """,
                unsafe_allow_html=True
            )

def prediction_page(model):
    """Página de predicción"""
    st.header("Predicción de Radiografías")
    
    if model is None:
        st.error("El modelo no está disponible. Problema de memoria.")
        st.info("""
        Posibles soluciones:
        1. El modelo es demasiado grande para el plan gratuito
        2. Intenta actualizar a un plan con más memoria
        3. Contacta al administrador del sistema
        """)
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **Sube una Radiografía:**
        - Formatos: JPG, PNG, JPEG
        - Tamaño recomendado: 224x224 píxeles
        - Radiografía pulmonar frontal
        """)
    
    with col2:
        st.warning("""
        **Advertencia:**
        Sistema para investigación.
        Validar con radiólogo.
        No usar para diagnóstico clínico.
        """)
    
    st.markdown("---")
    
    # Upload de imagen
    uploaded_file = st.file_uploader(
        "Selecciona una radiografía pulmonar",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        try:
            # Limpiar memoria antes de procesar
            gc.collect()
            
            # Cargar y mostrar imagen
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen Subida")
                st.image(image, use_column_width=True)
                
                # Información básica
                with st.expander("Detalles de imagen"):
                    st.write(f"Tamaño: {image.size[0]} x {image.size[1]}")
                    st.write(f"Formato: {uploaded_file.name.split('.')[-1].upper()}")
            
            with col2:
                st.subheader("Análisis")
                
                with st.spinner("Procesando imagen..."):
                    predicted_class, confidence, all_probs = predict_image(model, image)
                    
                    # Mostrar resultados
                    result_color = Config.CLASS_COLORS[Config.CLASS_NAMES.index(predicted_class)]
                    
                    st.markdown(
                        f"""
                        <div style='
                            background-color: {result_color}15;
                            border: 1px solid {result_color};
                            border-radius: 8px;
                            padding: 15px;
                            text-align: center;
                            margin: 10px 0;
                        '>
                        <h3 style='color: {result_color}; margin: 0;'>{predicted_class}</h3>
                        <p style='margin: 5px 0;'>Confianza: <strong>{confidence:.1%}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Gráfico simple
                    if all_probs:
                        fig = create_probability_chart(all_probs)
                        if fig:
                            st.pyplot(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Detalles
            st.subheader("Detalles de la Predicción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Probabilidades:**")
                for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"- {cls}: {prob:.1%}")
            
            with col2:
                st.write("**Descripción:**")
                st.info(Config.CLASS_DESCRIPTIONS[predicted_class])
                st.write("**Recomendación:**")
                st.warning(Config.CLASS_RECOMMENDATIONS[predicted_class])
            
            # Exportar simple
            st.markdown("---")
            if st.button("Generar Reporte Simple"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                report = f"""Predicción COVID-19 - {timestamp}
Clase: {predicted_class}
Confianza: {confidence:.1%}
Archivo: {uploaded_file.name}
"""
                st.download_button(
                    "Descargar Reporte",
                    report,
                    f"reporte_{timestamp.replace(':', '-')}.txt"
                )
            
            # Limpiar después de procesar
            del image
            gc.collect()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Intenta con otra imagen o contacta soporte.")
    else:
        st.info("Sube una imagen para comenzar el análisis.")

def analysis_page(history, val_acc):
    """Página de análisis del modelo"""
    st.header("Análisis del Modelo")
    
    st.subheader("Resumen del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Arquitectura", "ViT-Base")
    
    with col2:
        st.metric("Parámetros", "86M")
    
    with col3:
        st.metric("Input Size", "224x224")
    
    with col4:
        st.metric("Pre-entrenado", "ImageNet")
    
    st.markdown("---")
    
    # Historial de entrenamiento
    if history:
        st.subheader("Historial de Entrenamiento")
        fig = create_training_history_plot(history)
        if fig:
            st.pyplot(fig, use_container_width=True)
    
    # Información técnica
    st.subheader("Información Técnica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Hiperparámetros:**")
        st.write("- Learning Rate: 2e-4")
        st.write("- Batch Size: 32")
        st.write("- Optimizador: AdamW")
    
    with col2:
        st.write("**Preprocesamiento:**")
        st.write("- Resize: 224x224")
        st.write("- Normalización: ImageNet")
        st.write("- Split: 80/20")

def info_page():
    """Página de información"""
    st.header("Información del Sistema")
    
    st.info("""
    **Objetivo del Sistema:**
    Herramienta de IA para asistir en detección de condiciones pulmonares.
    No es sistema de diagnóstico automático.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Base de Datos:**")
        st.write("- COVID-19 Radiography Database")
        st.write("- 21,165 imágenes totales")
        st.write("- 4 clases balanceadas")
    
    with col2:
        st.write("**Arquitectura:**")
        st.write("- Vision Transformer (ViT)")
        st.write("- 12 capas transformer")
        st.write("- 12 heads de atención")
    
    st.markdown("---")
    
    st.warning("""
    **Limitaciones:**
    - Solo radiografías frontales
    - Sensible a calidad de imagen
    - Posibles falsos positivos/negativos
    - Para investigación únicamente
    """)

# ==================== APLICACIÓN PRINCIPAL ====================
def main():
    """Función principal de la aplicación Streamlit"""
    
    # Configuración básica de la página
    st.set_page_config(
        page_title="Sistema COVID-19 - Vision Transformer",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS minimalista
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Cargar modelo con spinner
    with st.spinner("Inicializando sistema..."):
        model, history, val_acc = load_model()
    
    # Crear UI
    create_header()
    page = create_sidebar()
    
    # Navegación de páginas
    if page == "Inicio":
        home_page(model, history, val_acc)
    elif page == "Predicción":
        prediction_page(model)
    elif page == "Análisis":
        analysis_page(history, val_acc)
    elif page == "Información":
        info_page()
    
    # Pie de página
    create_footer()
    
    # Limpieza final
    gc.collect()

if __name__ == "__main__":
    main()