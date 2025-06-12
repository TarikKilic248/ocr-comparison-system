import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import torch.cuda.amp as amp
import time
import json
import random

# PyTorch versiyon kontrolü
TORCH_VERSION = tuple(map(int, torch.__version__.split('.')[:2]))
USE_NEW_AMP = TORCH_VERSION >= (1, 10)

print(f"🔍 PyTorch Version: {torch.__version__}")
print(f"⚡ AMP API: {'New' if USE_NEW_AMP else 'Legacy'}")

class OCRDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_length=50, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.max_length = max_length
        self.augment = augment
        
        # Görüntü ve etiket dosyalarını eşleştir
        self.samples = []
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(data_dir, filename)
                    txt_path = os.path.join(data_dir, filename.replace('.jpg', '.txt'))
                    
                    if os.path.exists(txt_path):
                        self.samples.append((img_path, txt_path))
        
        if not self.samples:
            print(f"⚠️  Uyarı: {data_dir} dizininde veri bulunamadı!")
            return
        
        # Data augmentation ile veri artırma
        if augment:
            self._augment_data()
        
        # Karakter eşleştirmesini oluştur
        self.char_to_idx = self._create_char_mapping()
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        print(f"📊 Veri seti: {len(self.samples)} örnek, {len(self.char_to_idx)} karakter")
        
    def _augment_data(self):
        """Veri artırma - daha fazla eğitim verisi oluştur"""
        original_samples = self.samples.copy()
        augmented_samples = []
        
        # Her orijinal veri için 3 farklı versiyon oluştur
        for img_path, txt_path in original_samples:
            for i in range(3):
                # Augmented dosya adları
                base_name = os.path.splitext(img_path)[0]
                aug_img_path = f"{base_name}_aug_{i}.jpg"
                aug_txt_path = f"{base_name}_aug_{i}.txt"
                
                # Augmented görüntü oluştur
                try:
                    original_img = Image.open(img_path)
                    aug_img = self._apply_augmentation(original_img)
                    aug_img.save(aug_img_path, 'JPEG', quality=95)
                    
                    # Metin dosyasını kopyala
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    with open(aug_txt_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    augmented_samples.append((aug_img_path, aug_txt_path))
                except Exception as e:
                    print(f"Augmentation hatası: {e}")
                    continue
        
        self.samples.extend(augmented_samples)
        print(f"📈 Data augmentation: {len(original_samples)} → {len(self.samples)} örnek")
    
    def _apply_augmentation(self, image):
        """Görüntü augmentation uygula"""
        # Rastgele augmentation seç
        augmentations = []
        
        # Döndürme (-5 ile +5 derece)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor='white')
        
        # Kontrast değişimi
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # Parlaklık değişimi
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
        
        # Hafif bulanıklaştırma
        if random.random() > 0.7:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Keskinlik
        if random.random() > 0.7:
            enhancer = ImageEnhance.Sharpness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        return image
    
    def _create_char_mapping(self):
        """Karakter mapping oluştur"""
        all_chars = set('<BLANK>')
        
        for _, txt_path in self.samples:
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    all_chars.update(text)
            except Exception as e:
                continue
        
        # Temel karakterler
        basic_chars = 'abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ0123456789.,;:!?()[]{}@#$%^&*+-=<>/\\|_~`"\' '
        all_chars.update(basic_chars)
        
        sorted_chars = ['<BLANK>'] + sorted([c for c in all_chars if c != '<BLANK>'])
        return {char: idx for idx, char in enumerate(sorted_chars)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]
        
        # Görüntü yükle
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            image = torch.zeros(3, 224, 224)  # Daha küçük boyut
        
        # Metin yükle
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            text = ""
        
        # Metin temizle ve kısalt
        text = text[:self.max_length]
        
        # Encoding
        label_indices = [self.char_to_idx.get(c, 1) for c in text]
        label_length = len(label_indices)
        
        return image, torch.tensor(label_indices, dtype=torch.long), torch.tensor(label_length, dtype=torch.long)

class CompactOCRCNN(nn.Module):
    """Daha küçük ve efficient model"""
    def __init__(self, num_classes, max_length=50):
        super(CompactOCRCNN, self).__init__()
        self.max_length = max_length
        
        # Daha küçük CNN
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
        features = self.features(x)  # [batch, 256, 1, 1]
        features = features.view(batch_size, 256)  # [batch, 256]
        
        # Sequence oluştur
        features = features.unsqueeze(1).repeat(1, self.max_length, 1)  # [batch, seq, 256]
        
        # RNN
        rnn_out, _ = self.rnn(features)  # [batch, seq, 256]
        rnn_out = self.dropout(rnn_out)
        
        # Çıkış
        output = self.classifier(rnn_out)  # [batch, seq, num_classes]
        output = output.transpose(0, 1)  # [seq, batch, num_classes] - CTC için
        
        return output

class EarlyStopping:
    """Early stopping callback"""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def get_autocast_context(device):
    """PyTorch versiyonuna göre doğru autocast context döndür"""
    if not device.type == 'cuda':
        return None
    
    if USE_NEW_AMP:
        try:
            return amp.autocast(device_type='cuda')
        except:
            return amp.autocast()
    else:
        return amp.autocast()

def get_grad_scaler(device):
    """PyTorch versiyonuna göre doğru GradScaler döndür"""
    if not device.type == 'cuda':
        return None
    
    if USE_NEW_AMP:
        try:
            return amp.GradScaler(device='cuda')
        except:
            return amp.GradScaler()
    else:
        return amp.GradScaler()

def train_model_improved(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """İyileştirilmiş eğitim fonksiyonu - Tüm PyTorch versiyonları uyumlu"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)
    
    # Mixed precision - versiyon uyumlu
    scaler = get_grad_scaler(device)
    use_amp = scaler is not None
    
    print(f"🚀 İyileştirilmiş eğitim başlıyor:")
    print(f"   📊 Epochs: {num_epochs}")
    print(f"   📊 Train batches: {len(train_loader)}")
    print(f"   📊 Val batches: {len(val_loader)}")
    print(f"   🖥️  Device: {device}")
    print(f"   ⚡ Mixed Precision: {use_amp}")
    print(f"   🛑 Early Stopping: Patience={early_stopping.patience}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # === TRAINING ===
        model.train()
        train_loss = 0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [TRAIN]')
        for batch_idx, (images, labels, label_lengths) in enumerate(train_pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision training - versiyon uyumlu
            if use_amp:
                autocast_context = get_autocast_context(device)
                with autocast_context:
                    outputs = model(images)
                    log_probs = outputs.log_softmax(2)
                    input_lengths = torch.full((images.size(0),), outputs.size(0), 
                                             dtype=torch.long, device=device)
                    loss = criterion(log_probs, labels, input_lengths, label_lengths)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                log_probs = outputs.log_softmax(2)
                input_lengths = torch.full((images.size(0),), outputs.size(0), 
                                         dtype=torch.long, device=device)
                loss = criterion(log_probs, labels, input_lengths, label_lengths)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{train_loss/train_batches:.4f}'
            })
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [VAL]')
            for images, labels, label_lengths in val_pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)
                
                try:
                    if use_amp:
                        autocast_context = get_autocast_context(device)
                        with autocast_context:
                            outputs = model(images)
                            log_probs = outputs.log_softmax(2)
                            input_lengths = torch.full((images.size(0),), outputs.size(0), 
                                                     dtype=torch.long, device=device)
                            loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    else:
                        outputs = model(images)
                        log_probs = outputs.log_softmax(2)
                        input_lengths = torch.full((images.size(0),), outputs.size(0), 
                                                 dtype=torch.long, device=device)
                        loss = criterion(log_probs, labels, input_lengths, label_lengths)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                except Exception as e:
                    print(f"Val batch hatası: {e}")
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Learning rate schedule
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # En iyi model kaydet
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, 'best_model_checkpoint.pth')
            print(f"💾 En iyi model kaydedildi (val_loss: {best_val_loss:.4f})")
        
        # Epoch özeti
        epoch_time = time.time() - start_time
        improvement = "📈" if len(val_losses) == 1 or val_losses[-1] < val_losses[-2] else "📉"
        
        print(f'📊 Epoch {epoch+1}/{num_epochs}:')
        print(f'   🎯 Train Loss: {avg_train_loss:.4f}')
        print(f'   🎯 Val Loss: {avg_val_loss:.4f} {improvement}')
        print(f'   📈 Learning Rate: {current_lr:.6f}')
        print(f'   ⏱️  Süre: {epoch_time:.1f}s')
        
        # Early stopping kontrolü
        if early_stopping(avg_val_loss):
            print(f"🛑 Early stopping! {early_stopping.patience} epoch boyunca iyileşme yok.")
            break
        
        print("-" * 60)
        
        # GPU memory temizle
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return train_losses, val_losses

def ocr_collate_fn(batch):
    """CTC için batch hazırlama"""
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, labels, label_lengths

def main():
    print("🤖 İyileştirilmiş OCR Model Eğitimi (Uyumlu Versiyon)")
    print("=" * 60)
    
    # GPU kontrolü
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("⚠️  GPU bulunamadı, CPU kullanılacak!")
    
    print(f"🎯 Device: {device}")
    print("-" * 60)
    
    # Geliştirilmiş transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Daha küçük boyut
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Veri setleri - Augmentation ile
    print("📂 Veri setleri yükleniyor...")
    train_dataset = OCRDataset('train_data', transform=train_transform, max_length=50, augment=True)
    val_dataset = OCRDataset('val_data', transform=val_transform, max_length=50, augment=False)
    
    if len(train_dataset) == 0:
        print("❌ Eğitim verisi bulunamadı!")
        return
    
    # DataLoader - Küçük batch size
    batch_size = 8 if device.type == 'cuda' else 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=ocr_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=ocr_collate_fn
    )
    
    print(f"📊 Eğitim örnekleri: {len(train_dataset)} (augmented)")
    print(f"📊 Doğrulama örnekleri: {len(val_dataset)}")
    print(f"📊 Karakter sayısı: {len(train_dataset.char_to_idx)}")
    print(f"📊 Batch size: {batch_size}")
    
    # Compact model
    print("\n🏗️  Compact model oluşturuluyor...")
    model = CompactOCRCNN(num_classes=len(train_dataset.char_to_idx), max_length=50)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Model parametreleri: {total_params:,}")
    
    # Optimizer - Daha düşük learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
    )
    
    # Eğitim
    print("\n🚀 Eğitim başlıyor...")
    train_losses, val_losses = train_model_improved(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device
    )
    
    # Model kaydet
    print("\n💾 Final model kaydediliyor...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': train_dataset.char_to_idx,
        'idx_to_char': train_dataset.idx_to_char,
        'max_length': train_dataset.max_length,
        'num_classes': len(train_dataset.char_to_idx),
        'model_architecture': 'CompactOCRCNN',
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'ocr_model.pth')
    
    # Grafik oluştur
    if len(train_losses) > 0:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        if len(train_losses) > 10:
            # Moving average
            window = 5
            train_ma = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            val_ma = np.convolve(val_losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(train_losses)), train_ma, label='Train MA', linewidth=2)
            plt.plot(range(window-1, len(val_losses)), val_ma, label='Val MA', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (Moving Average)')
            plt.title('Smoothed Loss Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        # Overfitting analizi
        overfitting_gap = [val - train for val, train in zip(val_losses, train_losses)]
        plt.plot(overfitting_gap, label='Val - Train Loss', linewidth=2, color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Gap')
        plt.title('Overfitting Analysis')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Learning rate etkisi
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, val_losses, 'o-', label='Validation Loss', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_training_results.png', dpi=300, bbox_inches='tight')
        
        print("📈 Eğitim grafikleri 'improved_training_results.png' dosyasına kaydedildi.")
    
    print("\n🎉 İyileştirilmiş eğitim tamamlandı!")
    print(f"📂 Kaydedilen dosyalar:")
    print(f"   📄 ocr_model.pth - Final model")
    print(f"   📄 best_model_checkpoint.pth - En iyi model")
    print(f"   📄 improved_training_results.png - Detaylı analiz")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"🧹 GPU memory temizlendi.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Eğitim iptal edildi.")
    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")
        import traceback
        traceback.print_exc()