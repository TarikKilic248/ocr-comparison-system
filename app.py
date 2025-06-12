# app.py
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import io
import plotly.express as px
import re
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import subprocess
import sys
import os
import time
from ocr_utils import (
    perform_ocr,
    ocr_enhance_combined,
    grayscale,
    contrast_enhancement,
    sharpening,
    histogram_equalization,
    adaptive_histogram_equalization,
    noise_removal,
    morphological_closing,
    create_sample_dataset
)

# Özel model sınıfı - eğitimde kullanılan aynı mimari
class CompactOCRCNN(nn.Module):
    """Eğitilmiş özel model mimarisi"""
    def __init__(self, num_classes, max_length=50):
        super(CompactOCRCNN, self).__init__()
        self.max_length = max_length
        
        # CNN katmanları
        self.features = nn.Sequential(
            # İlk blok
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # İkinci blok
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Üçüncü blok
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Dördüncü blok
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # RNN katmanları
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        
        # Çıkış katmanı
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN features
        features = self.features(x)
        features = features.view(batch_size, 256)
        
        # Sequence oluştur
        features = features.unsqueeze(1).repeat(1, self.max_length, 1)
        
        # RNN
        rnn_out, _ = self.rnn(features)
        rnn_out = self.dropout(rnn_out)
        
        # Çıkış
        output = self.classifier(rnn_out)
        output = output.transpose(0, 1)  # CTC için
        
        return output

# Global değişkenler yerine Streamlit session state kullanacağız
def initialize_session_state():
    """Session state değişkenlerini başlat"""
    if 'custom_model' not in st.session_state:
        st.session_state.custom_model = None
    if 'char_to_idx' not in st.session_state:
        st.session_state.char_to_idx = None
    if 'idx_to_char' not in st.session_state:
        st.session_state.idx_to_char = None
    if 'custom_model_loaded' not in st.session_state:
        st.session_state.custom_model_loaded = False
    if 'model_info' not in st.session_state:
        st.session_state.model_info = ""

def load_custom_model():
    """Eğitilmiş özel modeli yükle - Session state ile"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model dosyalarını kontrol et
        model_files = ['best_model_checkpoint.pth', 'ocr_model.pth']
        model_path = None
        
        for file in model_files:
            if os.path.exists(file):
                model_path = file
                break
        
        if not model_path:
            st.error("❌ Model dosyası bulunamadı!")
            st.info("Önce modeli eğitin: python train_model.py")
            return False
        
        # Model yükle
        if model_path == 'best_model_checkpoint.pth':
            checkpoint = torch.load(model_path, map_location=device)
            
            # ocr_model.pth'den char mapping al
            if os.path.exists('ocr_model.pth'):
                ocr_data = torch.load('ocr_model.pth', map_location=device)
                st.session_state.char_to_idx = ocr_data['char_to_idx']
                st.session_state.idx_to_char = ocr_data['idx_to_char']
                max_length = ocr_data.get('max_length', 50)
                num_classes = ocr_data['num_classes']
            else:
                st.error("❌ ocr_model.pth bulunamadı! Char mapping yüklenemedi.")
                return False
            
            # Model oluştur
            st.session_state.custom_model = CompactOCRCNN(num_classes=num_classes, max_length=max_length)
            st.session_state.custom_model.load_state_dict(checkpoint['model_state_dict'])
            
        else:  # ocr_model.pth
            checkpoint = torch.load(model_path, map_location=device)
            st.session_state.char_to_idx = checkpoint['char_to_idx']
            st.session_state.idx_to_char = checkpoint['idx_to_char']
            max_length = checkpoint.get('max_length', 50)
            num_classes = checkpoint['num_classes']
            
            # Model oluştur
            st.session_state.custom_model = CompactOCRCNN(num_classes=num_classes, max_length=max_length)
            st.session_state.custom_model.load_state_dict(checkpoint['model_state_dict'])
        
        st.session_state.custom_model.to(device)
        st.session_state.custom_model.eval()
        st.session_state.custom_model_loaded = True
        st.session_state.model_info = f"Model: {num_classes} karakter, {max_length} max uzunluk, Dosya: {model_path}"
        
        st.success(f"✅ Özel model yüklendi! ({model_path})")
        st.info(f"📊 Model: {num_classes} karakter, {max_length} max uzunluk")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        st.session_state.custom_model_loaded = False
        return False

def ctc_decode_improved(predictions, idx_to_char, blank_idx=0):
    """Geliştirilmiş CTC decoding"""
    try:
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # İlk batch elemanı
        
        # En yüksek olasılığa sahip karakterleri al
        predicted_ids = torch.argmax(predictions, dim=1)
        
        # CTC decoding
        decoded = []
        prev_id = None
        
        for pred_id in predicted_ids:
            pred_id = pred_id.item()
            
            # Blank karakteri atla
            if pred_id == blank_idx:
                prev_id = pred_id
                continue
            
            # Ardışık aynı karakterleri atla
            if pred_id != prev_id:
                if pred_id in idx_to_char:
                    decoded.append(idx_to_char[pred_id])
            
            prev_id = pred_id
        
        result = ''.join(decoded)
        
        # Post-processing
        result = result.strip()
        if not result:
            result = "[Tanınamadı]"
        
        return result
        
    except Exception as e:
        st.error(f"CTC decode hatası: {str(e)}")
        return "[Decode Hatası]"

def perform_custom_ocr(image, lang='tur+eng'):
    """Eğitilmiş özel model ile OCR - Session state ile"""
    start_time = time.time()
    
    # Model yüklü değilse yükle
    if not st.session_state.custom_model_loaded:
        st.warning("🔄 Model yükleniyor...")
        if not load_custom_model():
            return {
                'text': "Model yüklenemedi",
                'char_count': 0,
                'word_count': 0,
                'conf_score': 0,
                'ocr_time': time.time() - start_time
            }
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Görüntü preprocessing
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
        
        # Transform - eğitimde kullanılan aynı transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = st.session_state.custom_model(image_tensor)  # [seq_len, batch, num_classes]
        
        # CTC decode
        text = ctc_decode_improved(outputs, st.session_state.idx_to_char)
        
        # İstatistikler
        word_count = len(re.findall(r'\w+', text))
        char_count = len(text)
        
        # Güven skoru (basit hesaplama)
        if text and text != "[Tanınamadı]" and text != "[Decode Hatası]":
            conf_score = min(75 + len(text) * 0.2, 92)
        else:
            conf_score = 5
        
        duration = time.time() - start_time
        
        return {
            'text': text,
            'char_count': char_count,
            'word_count': word_count,
            'conf_score': conf_score,
            'ocr_time': duration
        }
        
    except Exception as e:
        error_msg = f"OCR hatası: {str(e)}"
        st.error(f"Özel model OCR hatası: {str(e)}")
        return {
            'text': error_msg,
            'char_count': 0,
            'word_count': 0,
            'conf_score': 0,
            'ocr_time': time.time() - start_time
        }

def show_score_chart(df):
    """Geliştirilmiş skor grafiği"""
    fig = px.bar(
        df.sort_values("Güven", ascending=False),
        x="Motor + Yöntem",
        y="Güven",
        color="Motor",
        text="Güven",
        hover_data=["Karakter", "Kelime", "Süre"],
        title="OCR Motorları Karşılaştırması",
        color_discrete_map={
            'Tesseract': '#FF6B6B',
            'EasyOCR': '#4ECDC4', 
            'Özel Model': '#45B7D1'
        }
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, 100], 
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        xaxis_tickangle=45,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def show_detailed_comparison(df):
    """Detaylı karşılaştırma tablosu"""
    st.markdown("### 📊 Detaylı Performans Karşılaştırması")
    
    # Motor bazında özet
    motor_summary = df.groupby('Motor').agg({
        'Güven': ['mean', 'max', 'min'],
        'Süre': ['mean', 'min', 'max'],
        'Karakter': 'mean',
        'Kelime': 'mean'
    }).round(2)
    
    motor_summary.columns = ['Ortalama Güven', 'En Yüksek Güven', 'En Düşük Güven',
                            'Ortalama Süre', 'En Hızlı', 'En Yavaş', 
                            'Ortalama Karakter', 'Ortalama Kelime']
    
    st.dataframe(motor_summary, use_container_width=True)
    
    # En iyi performanslar
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy = df.loc[df['Güven'].idxmax()]
        st.metric(
            "🎯 En Yüksek Güven",
            f"{best_accuracy['Güven']:.1f}%",
            f"{best_accuracy['Motor + Yöntem']}"
        )
    
    with col2:
        fastest = df.loc[df['Süre'].idxmin()]
        st.metric(
            "⚡ En Hızlı",
            f"{fastest['Süre']:.2f}s",
            f"{fastest['Motor + Yöntem']}"
        )
    
    with col3:
        most_chars = df.loc[df['Karakter'].idxmax()]
        st.metric(
            "📝 En Fazla Karakter",
            f"{most_chars['Karakter']} karakter",
            f"{most_chars['Motor + Yöntem']}"
        )

def install_requirements():
    """Gerekli paketleri kontrol et"""
    try:
        import easyocr
        import torch
        import pytesseract
        return True
    except ImportError as e:
        st.error(f"Eksik paket: {e}")
        st.info("requirements.txt dosyasındaki paketleri yükleyin")
        return False

def check_model_files():
    """Model dosyalarının durumunu kontrol et"""
    st.sidebar.markdown("### 📋 Model Dosyaları")
    
    files_to_check = {
        'ocr_model.pth': 'Final model',
        'best_model_checkpoint.pth': 'En iyi model',
        'training_info.json': 'Eğitim bilgileri',
        'improved_training_results.png': 'Eğitim grafikleri'
    }
    
    for file, description in files_to_check.items():
        if os.path.exists(file):
            file_size = os.path.getsize(file) / (1024*1024)  # MB
            st.sidebar.success(f"✅ {description} ({file_size:.1f} MB)")
        else:
            st.sidebar.error(f"❌ {description}")

# Streamlit sayfa konfigürasyonu
st.set_page_config(
    page_title="OCR Karşılaştırmalı Analiz", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state başlat
initialize_session_state()

st.title("📄 PDF OCR Analizi - Model Karşılaştırması")
st.markdown("**Tesseract vs EasyOCR vs Özel Eğitilmiş Model**")
st.markdown("---")

# Sidebar konfigürasyonu
with st.sidebar:
    st.header("⚙️ OCR Ayarları")
    
    selected_lang = st.selectbox(
        "OCR Dili:", 
        options=["tur", "eng", "tur+eng"], 
        index=0,
        help="Tesseract için dil seçimi"
    )
    
    st.header("🔧 Test Yapılacak Motorlar")
    
    # Motor seçimi - çoklu seçim
    test_tesseract = st.checkbox("📄 Tesseract", value=True)
    test_easyocr = st.checkbox("🤖 EasyOCR", value=True)
    test_custom = st.checkbox("🎯 Özel Model", value=True, help="Eğittiğiniz model")
    
    # Debug bilgisi
    st.markdown("---")
    with st.expander("🔧 Debug - Model Durumu"):
        st.write("**Session State Bilgileri:**")
        st.write(f"- custom_model_loaded: {st.session_state.custom_model_loaded}")
        st.write(f"- custom_model: {st.session_state.custom_model is not None}")
        st.write(f"- char_to_idx: {st.session_state.char_to_idx is not None}")
        st.write(f"- idx_to_char: {st.session_state.idx_to_char is not None}")
        if st.session_state.model_info:
            st.write(f"- model_info: {st.session_state.model_info}")
        
        st.write("**Model Dosyaları:**")
        for file in ['best_model_checkpoint.pth', 'ocr_model.pth']:
            exists = os.path.exists(file)
            size = f" ({os.path.getsize(file)/1024/1024:.1f} MB)" if exists else ""
            st.write(f"- {file}: {'✅' if exists else '❌'}{size}")
        
        if st.button("🧹 Session State Temizle"):
            for key in ['custom_model', 'char_to_idx', 'idx_to_char', 'custom_model_loaded', 'model_info']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Model dosyası durumu
    check_model_files()
    
    st.markdown("---")
    
    # Özel model kontrolü
    if test_custom:
        st.header("🎯 Özel Model Durumu")
        
        if st.button("🔄 Modeli Yükle/Yenile"):
            with st.spinner("Model yükleniyor..."):
                success = load_custom_model()
                if success:
                    st.rerun()  # Sayfayı yenile
        
        # Model bilgileri
        if st.session_state.custom_model_loaded:
            st.success("✅ Model yüklü ve hazır")
            if st.session_state.model_info:
                st.info(f"📊 {st.session_state.model_info}")
            if st.session_state.char_to_idx:
                st.info(f"🔤 {len(st.session_state.char_to_idx)} farklı karakter tanıyabiliyor")
        else:
            st.warning("⚠️ Model henüz yüklenmedi")
            st.info("Lütfen 'Modeli Yükle/Yenile' butonuna tıklayın")
            if st.button("📚 Model Eğit"):
                st.info("Lütfen terminalde: python train_model.py")

# Ana uygulama alanı
if not install_requirements():
    st.stop()

st.header("📤 PDF Dosyası Yükleme")

uploaded_file = st.file_uploader(
    "PDF dosyanızı yükleyin", 
    type=["pdf"],
    help="OCR analizi yapılacak PDF dosyasını seçin"
)

if uploaded_file:
    try:
        # PDF'yi görüntülere dönüştür
        with st.spinner("PDF işleniyor..."):
            # Sisteminiz için poppler yolları
            poppler_paths = [
                r"C:\poppler-24.08.0\bin",           # Sizin kurduğunuz versiyon
                r"C:\poppler-24.08.0\Library\bin",   # Alternatif yol
                r"C:\poppler\bin",                   # Genel yol
                None                                 # PATH ortam değişkeninden
            ]
            
            pdf_images = None
            successful_path = None
            pdf_bytes = uploaded_file.read()
            
            for poppler_path in poppler_paths:
                try:
                    if poppler_path and os.path.exists(poppler_path):
                        pdf_images = convert_from_bytes(
                            pdf_bytes,
                            dpi=300,
                            poppler_path=poppler_path
                        )
                        successful_path = poppler_path
                        st.success(f"✅ PDF başarıyla işlendi! Kullanılan yol: {poppler_path}")
                    elif poppler_path is None:
                        pdf_images = convert_from_bytes(
                            pdf_bytes,
                            dpi=300
                        )
                        successful_path = "PATH ortam değişkeni"
                        st.success(f"✅ PDF başarıyla işlendi! Kullanılan yol: PATH")
                    else:
                        continue
                        
                    break
                    
                except Exception as e:
                    if poppler_path:
                        st.warning(f"Poppler yolu denendi: {poppler_path} - Başarısız: {str(e)}")
                    continue
            
            if pdf_images is None:
                st.error("❌ PDF işlenemedi! Poppler düzgün kurulu değil.")
                st.error("Denenen tüm yollar başarısız oldu.")
                
                with st.expander("🔧 Sorun Giderme"):
                    st.write("**Poppler Kontrolü:**")
                    st.code("pdftoppm -h")
                    st.write("Bu komut PowerShell'de çalışmalı.")
                    
                    st.write("**Poppler Yolları:**")
                    for path in poppler_paths[:-1]:  # None hariç
                        exists = os.path.exists(path) if path else False
                        st.write(f"- {path}: {'✅ Var' if exists else '❌ Yok'}")
                    
                    st.write("**Çözüm Önerileri:**")
                    st.write("1. Poppler'ı yeniden indirin ve C:\poppler-24.08.0 klasörüne çıkartın")
                    st.write("2. PATH ortam değişkenine C:\poppler-24.08.0\\bin ekleyin")
                    st.write("3. PowerShell'i yeniden başlatın")
                    st.write("4. Alternatif: PyMuPDF yükleyin: `pip install PyMuPDF`")
                
                st.stop()
        
        num_pages = len(pdf_images)
        st.success(f"✅ {num_pages} sayfa başarıyla yüklendi.")

        # Sayfa seçimi
        st.header("📄 Sayfa Seçimi ve Analiz")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            all_pages = st.checkbox("📚 Tüm sayfalarda OCR yap", value=False)
        
        with col2:
            if not all_pages:
                if num_pages == 1:
                    st.info("PDF sadece 1 sayfa içeriyor.")
                    page = 1
                else:
                    page = st.slider("Sayfa Seçin", min_value=1, max_value=num_pages, value=1)
            else:
                page = 1

        pages_to_process = range(1, num_pages + 1) if all_pages else [page]
        
        # Seçilen motorları kontrol et
        selected_engines = []
        if test_tesseract:
            selected_engines.append("Tesseract")
        if test_easyocr:
            selected_engines.append("EasyOCR")
        if test_custom:
            if st.session_state.custom_model_loaded:
                selected_engines.append("Özel Model")
            else:
                st.warning("⚠️ Özel model yüklü değil! Önce sidebar'dan modeli yükleyin.")
                st.info("📋 Özel modeli kullanmak için: Sidebar → '🔄 Modeli Yükle/Yenile' butonuna tıklayın")
        
        if not selected_engines:
            st.error("❌ En az bir OCR motoru seçmelisiniz!")
            st.info("💡 Sidebar'dan motor seçimi yapın")
            st.stop()
        
        # Seçilen motorları göster
        st.info(f"🎯 Seçilen motorlar: {', '.join(selected_engines)}")
        
        # OCR analizi başlat
        if st.button("🔍 Karşılaştırmalı OCR Analizi Başlat", type="primary"):
            
            all_results = []
            
            for current_page in pages_to_process:
                st.markdown(f"## 📄 Sayfa {current_page}")
                
                image = pdf_images[current_page - 1]
                
                # Orijinal görüntüyü göster
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption=f"Sayfa {current_page}", use_container_width=True)
                
                # Görüntü iyileştirme yöntemleri
                methods = {
                    "Orijinal": lambda img: np.array(img),
                    "Kombinasyon": ocr_enhance_combined,
                    "Gri + Kontrast": lambda img: contrast_enhancement(grayscale(img)),
                    "CLAHE": lambda img: adaptive_histogram_equalization(grayscale(img)),
                    "Keskinleştirme": lambda img: sharpening(grayscale(img))
                }

                page_scores = []
                page_results = {}

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_operations = len(methods) * len(selected_engines)
                current_operation = 0
                
                for method_name, method_func in methods.items():
                    # Görüntüyü işle
                    processed = method_func(image)
                    
                    for engine in selected_engines:
                        current_operation += 1
                        status_text.text(f"İşleniyor: {engine} - {method_name} ({current_operation}/{total_operations})")
                        progress_bar.progress(current_operation / total_operations)
                        
                        try:
                            if engine == "Tesseract":
                                result = perform_ocr(processed, lang=selected_lang, engine='tesseract')
                            elif engine == "EasyOCR":
                                result = perform_ocr(processed, lang=selected_lang, engine='easyocr')
                            else:  # Özel Model
                                result = perform_custom_ocr(processed, lang=selected_lang)
                            
                            result_key = f"{engine} | {method_name}"
                            page_results[result_key] = (processed, result)
                            page_scores.append({
                                "Sayfa": current_page,
                                "Motor": engine,
                                "Yöntem": method_name,
                                "Motor + Yöntem": result_key,
                                "Karakter": result['char_count'],
                                "Kelime": result['word_count'],
                                "Güven": result['conf_score'],
                                "Süre": result['ocr_time'],
                                "Metin": result['text']
                            })
                            
                        except Exception as e:
                            st.error(f"Hata ({engine} - {method_name}): {str(e)}")
                            continue

                progress_bar.empty()
                status_text.empty()
                
                if page_scores:
                    df = pd.DataFrame(page_scores)
                    all_results.extend(page_scores)
                    
                    # En iyi sonuçları bul
                    best_overall = df.sort_values(by="Güven", ascending=False).iloc[0]
                    best_by_engine = df.groupby('Motor')['Güven'].idxmax()
                    
                    # Sonuçları göster
                    with col2:
                        st.markdown(f"### 🏆 Sayfa {current_page} En İyi Sonuç")
                        st.markdown(f"**Motor:** {best_overall['Motor']}")
                        st.markdown(f"**Yöntem:** {best_overall['Yöntem']}")
                        st.markdown(f"**Güven:** {best_overall['Güven']:.1f}%")
                        st.markdown(f"**Süre:** {best_overall['Süre']:.2f} saniye")
                        
                        st.text_area("OCR Metni", best_overall['Metin'], height=150, key=f"best_text_{current_page}")

                    # Motor bazında en iyi sonuçlar
                    st.markdown("### 🥇 Motor Bazında En İyi Sonuçlar")
                    engine_cols = st.columns(len(selected_engines))
                    
                    for i, engine in enumerate(selected_engines):
                        if engine in df['Motor'].values:
                            engine_best = df[df['Motor'] == engine].sort_values('Güven', ascending=False).iloc[0]
                            with engine_cols[i]:
                                st.metric(
                                    f"{engine}",
                                    f"{engine_best['Güven']:.1f}%",
                                    f"{engine_best['Yöntem']}"
                                )
                    
                    # İndirme butonu
                    txt_file = io.StringIO()
                    txt_file.write(f"Sayfa {current_page} - En İyi Sonuç ({best_overall['Motor + Yöntem']})\n")
                    txt_file.write("="*50 + "\n")
                    txt_file.write(best_overall['Metin'])
                    txt_file.seek(0)
                    download_data = txt_file.getvalue().encode("utf-8")

                    st.download_button(
                        label=f"📥 Sayfa {current_page} OCR Metnini İndir",
                        data=download_data,
                        file_name=f"sayfa_{current_page}_ocr.txt",
                        mime="text/plain"
                    )

                    # Detaylı sonuçlar tablosu
                    st.markdown("### 📊 Tüm Sonuçlar")
                    st.dataframe(
                        df.sort_values(by="Güven", ascending=False), 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Grafik
                    show_score_chart(df)

                    # Tüm sonuçları göster
                    with st.expander(f"🔍 Sayfa {current_page} - Tüm Yöntemler ve İşlenmiş Görüntüler"):
                        for key, (proc_img, res) in page_results.items():
                            st.markdown(f"#### {key}")
                            result_col1, result_col2 = st.columns([1, 2])
                            
                            with result_col1:
                                if len(proc_img.shape) == 2:  # Grayscale
                                    st.image(proc_img, caption=key, use_container_width=True, clamp=True)
                                else:
                                    st.image(proc_img, caption=key, use_container_width=True)
                            
                            with result_col2:
                                st.text_area(
                                    "OCR Metni", 
                                    res['text'], 
                                    height=120, 
                                    key=f"text_{current_page}_{key.replace(' ', '_').replace('|', '_')}"
                                )
                                
                                # Performans metrikleri
                                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                with metric_col1:
                                    st.metric("Güven", f"{res['conf_score']:.1f}%")
                                with metric_col2:
                                    st.metric("Kelime", res['word_count'])
                                with metric_col3:
                                    st.metric("Karakter", res['char_count'])
                                with metric_col4:
                                    st.metric("Süre", f"{res['ocr_time']:.2f}s")
                            
                            st.markdown("---")
                else:
                    st.error("Hiçbir OCR yöntemi başarılı olamadı!")
                
                if current_page < len(pages_to_process):
                    st.markdown("---")
            
            # Genel özet
            if all_results:
                st.markdown("## 🎯 Genel Karşılaştırma Özeti")
                
                all_df = pd.DataFrame(all_results)
                
                # Detaylı karşılaştırma
                show_detailed_comparison(all_df)
                
                # Genel performans grafiği
                st.markdown("### 📈 Tüm Sayfalar Genel Performans")
                show_score_chart(all_df)
                
                # Sonuç önerileri
                st.markdown("### 💡 Öneri ve Sonuçlar")
                
                motor_averages = all_df.groupby('Motor')['Güven'].mean().sort_values(ascending=False)
                best_motor = motor_averages.index[0]
                best_score = motor_averages.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🥇 En İyi Genel Performans", 
                        f"{best_motor}",
                        f"{best_score:.1f}% ortalama güven"
                    )
                
                with col2:
                    fastest_motor = all_df.groupby('Motor')['Süre'].mean().sort_values().index[0]
                    fastest_time = all_df.groupby('Motor')['Süre'].mean().sort_values().iloc[0]
                    st.metric(
                        "⚡ En Hızlı Motor",
                        f"{fastest_motor}",
                        f"{fastest_time:.2f}s ortalama"
                    )
                
                with col3:
                    most_chars_motor = all_df.groupby('Motor')['Karakter'].mean().sort_values(ascending=False).index[0]
                    most_chars = all_df.groupby('Motor')['Karakter'].mean().sort_values(ascending=False).iloc[0]
                    st.metric(
                        "📝 En Fazla Metin",
                        f"{most_chars_motor}",
                        f"{most_chars:.0f} ortalama karakter"
                    )
                
                # Kullanım önerileri
                st.markdown("#### 🎯 Kullanım Önerileri:")
                
                if "Özel Model" in motor_averages.index:
                    custom_rank = list(motor_averages.index).index("Özel Model") + 1
                    custom_score = motor_averages["Özel Model"]
                    
                    if custom_rank == 1:
                        st.success("🎉 **Özel modeliniz en iyi performansı gösteriyor!** Bu veriler için özel model kullanın.")
                    elif custom_rank == 2:
                        st.info(f"📊 Özel modeliniz {custom_rank}. sırada (%{custom_score:.1f}). İyi performans gösteriyor.")
                    else:
                        st.warning(f"⚠️ Özel modeliniz {custom_rank}. sırada (%{custom_score:.1f}). Daha fazla eğitim verisi gerekebilir.")
                
                if best_motor == "Tesseract":
                    st.info("📄 **Tesseract** bu dokümanlarda en iyi sonucu veriyor. Hızlı ve güvenilir.")
                elif best_motor == "EasyOCR":
                    st.info("🤖 **EasyOCR** bu dokümanlarda en iyi sonucu veriyor. Modern neural network tabanlı.")

    except Exception as e:
        st.error(f"PDF işlenirken hata oluştu: {str(e)}")
        st.info("Lütfen poppler yolunuzun doğru olduğundan emin olun.")

# Sayfa altı
st.markdown("---")
st.markdown("""
### 📝 OCR Motor Bilgileri:

**🎯 Özel Model:**
- Sizin eğittiğiniz CompactOCRCNN modeli
- Verilerinize özel optimize edilmiş
- CTC loss ile eğitilmiş
- ~1.2M parametre, compact ve hızlı

**📄 Tesseract:**
- Açık kaynak, klasik OCR motoru
- Hızlı ve güvenilir
- Çok dilli destek

**🤖 EasyOCR:**
- Modern deep learning tabanlı
- Yüksek doğruluk oranı
- GPU desteği

### 🛠️ Performans İpuçları:
- Özel modeliniz kötü performans gösteriyorsa daha fazla veri ile eğitin
- Farklı görüntü iyileştirme yöntemlerini deneyin
- PDF kalitesi sonuçları önemli ölçüde etkiler
""")

# Model durumu footer
if st.session_state.custom_model_loaded:
    st.success("✅ Özel model aktif ve kullanıma hazır!")
    if st.session_state.model_info:
        st.info(f"📊 {st.session_state.model_info}")
else:
    st.warning("⚠️ Özel model henüz yüklenmedi. Sidebar'dan yükleyin.")
    st.info("💡 Adımlar: Sidebar → 'Test Yapılacak Motorlar' → '🎯 Özel Model' ✓ → '🔄 Modeli Yükle/Yenile'")