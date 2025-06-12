# ocr_utils.py
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import pytesseract
import re
import time
import os
import easyocr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms

# Tesseract dizini tanımı (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\batma\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\\Users\\batma\\AppData\\Local\\Programs\\Tesseract-OCR\\tessdata"

# EasyOCR okuyucu
reader = easyocr.Reader(['tr', 'en'])

# Global model değişkeni
global_custom_model = None
global_char_to_idx = None
global_idx_to_char = None

# Özel OCR Modeli için CNN sınıfı
class OCRNet(nn.Module):
    def __init__(self, num_classes=100, max_length=100):
        super(OCRNet, self).__init__()
        self.max_length = max_length
        
        # CNN katmanları - ResNet benzeri yapı
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual bloklar
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Tam bağlantılı katmanlar
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes * max_length)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), self.max_length, -1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = nn.ReLU(inplace=True)(out)
        
        return out

# Özel veri seti sınıfı
class OCRDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_custom_model(model_path='ocr_model.pth'):
    """Eğitilmiş özel modeli yükler"""
    global global_custom_model, global_char_to_idx, global_idx_to_char
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model dosyasını yükle
        checkpoint = torch.load(model_path, map_location=device)
        
        # Karakter mapping'i al
        global_char_to_idx = checkpoint['char_to_idx']
        global_idx_to_char = {v: k for k, v in global_char_to_idx.items()}
        max_length = checkpoint['max_length']
        
        # Modeli oluştur
        global_custom_model = OCRNet(num_classes=len(global_char_to_idx), max_length=max_length)
        global_custom_model.load_state_dict(checkpoint['model_state_dict'])
        global_custom_model.to(device)
        global_custom_model.eval()
        
        print(f"Model başarıyla yüklendi. Karakter sayısı: {len(global_char_to_idx)}")
        return True
        
    except Exception as e:
        print(f"Model yüklenirken hata: {str(e)}")
        return False

def ctc_decode(predictions, idx_to_char, blank_idx=0):
    """CTC çıktısını decode eder"""
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Batch'den ilk elemanı al
    
    # En yüksek olasılığa sahip karakterleri al
    predicted_ids = torch.argmax(predictions, dim=1)
    
    # CTC decoding: ardışık aynı karakterleri ve blank'leri kaldır
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
    
    return ''.join(decoded)

def perform_custom_ocr(image, lang='tur+eng'):
    """Özel model ile OCR yapar"""
    global global_custom_model, global_char_to_idx, global_idx_to_char
    
    start_time = time.time()
    
    # Model yüklü değilse yükle
    if global_custom_model is None:
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
        
        # Görüntüyü hazırla
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
        
        # Transform uygula
        transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = global_custom_model(image_tensor)
            
        # CTC decode
        text = ctc_decode(outputs, global_idx_to_char)
        
        # İstatistikleri hesapla
        word_count = len(re.findall(r'\w+', text))
        char_count = len(text)
        
        # Güven skorunu tahmin et (basit bir yaklaşım)
        conf_score = min(85 + len(text) * 0.5, 95)  # Model çıktısına göre ayarlanabilir
        
        duration = time.time() - start_time
        
        return {
            'text': text,
            'char_count': char_count,
            'word_count': word_count,
            'conf_score': conf_score,
            'ocr_time': duration
        }
        
    except Exception as e:
        print(f"Özel model OCR hatası: {str(e)}")
        return {
            'text': f"OCR hatası: {str(e)}",
            'char_count': 0,
            'word_count': 0,
            'conf_score': 0,
            'ocr_time': time.time() - start_time
        }

# Özel model eğitimi fonksiyonu
def train_custom_model(train_loader, model, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

def create_sample_dataset():
    """Örnek veri seti oluşturur (demo amaçlı)"""
    # Bu fonksiyon gerçek veri setiniz yoksa demo amaçlı kullanılabilir
    images = []
    labels = []
    
    # Basit renkli görüntüler oluştur
    for i in range(100):
        # 64x64 RGB görüntü
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        images.append(torch.tensor(img).permute(2, 0, 1).float() / 255.0)
        
        # Rastgele etiket
        label = torch.randint(0, 10, (10,))
        labels.append(label)
    
    return images, labels

# EasyOCR ile metin çıkarma
def perform_easyocr(image, lang=['tr', 'en']):
    start_time = time.time()
    results = reader.readtext(np.array(image))
    text = ' '.join([result[1] for result in results])
    duration = time.time() - start_time
    
    word_count = len(re.findall(r'\w+', text))
    char_count = len(text)
    
    # Güven skorunu hesapla
    conf_scores = [result[2] for result in results]
    conf_score = np.mean(conf_scores) * 100 if conf_scores else 0
    
    return {
        'text': text,
        'char_count': char_count,
        'word_count': word_count,
        'conf_score': conf_score,
        'ocr_time': duration
    }

def grayscale(image):
    image = np.array(image)
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def contrast_enhancement(image):
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced = enhancer.enhance(2.0)
    return np.array(enhanced)

def sharpening(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def histogram_equalization(image):
    gray = grayscale(image)
    return cv2.equalizeHist(gray)

def adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def morphological_closing(image, kernel_size=2):
    gray = grayscale(image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

def noise_removal(image):
    gray = grayscale(image)
    return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

def resize_image(image, scale=2.0):
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        width, height = image.size
        return image.resize((int(width * scale), int(height * scale)), Image.BICUBIC)

def ocr_enhance_combined(image):
    image = resize_image(image, scale=1.5)
    gray = grayscale(image)
    denoised = noise_removal(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    sharpened = sharpening(contrast)
    closed = morphological_closing(sharpened)
    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def perform_ocr(image, lang='tur+eng', config='--oem 3 --psm 6', engine='tesseract'):
    if engine == 'easyocr':
        return perform_easyocr(image, lang=['tr', 'en'])
    elif engine == 'custom':
        return perform_custom_ocr(image, lang=lang)
    
    # Tesseract OCR
    start_time = time.time()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    duration = time.time() - start_time
    word_count = len(re.findall(r'\w+', text))
    char_count = len(text)

    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT, config=config)
        conf_values = [int(conf) for conf in data['conf'] if conf != '-1']
        conf_score = np.mean(conf_values) if conf_values else 0
    except:
        conf_score = 0

    return {
        'text': text,
        'char_count': char_count,
        'word_count': word_count,
        'conf_score': conf_score,
        'ocr_time': duration
    }