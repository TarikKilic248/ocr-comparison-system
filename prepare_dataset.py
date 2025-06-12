import os
import shutil
from PIL import Image
import random
import tempfile
import sys
from pdf2image import convert_from_path

def create_dataset_structure():
    """Veri seti dizin yapısını oluşturur"""
    directories = ['raw_data', 'train_data', 'val_data', 'test_images']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory} dizini oluşturuldu/kontrol edildi.")

def convert_pdf_to_images(pdf_path, dpi=300):
    """PDF dosyasını görüntülere dönüştürür"""
    try:
        print(f"  📄 PDF işleniyor: {os.path.basename(pdf_path)}")
        
        # Sisteminiz için poppler yolları (sırasıyla denenecek)
        poppler_paths = [
            r"C:\poppler-24.08.0\bin",           # Ana kurulum yeri
            r"C:\poppler-24.08.0\Library\bin",   # Alternatif yol
            r"C:\poppler\bin",                   # Genel yol
            None                                 # PATH ortam değişkeninden
        ]
        
        images = None
        successful_path = None
        
        for poppler_path in poppler_paths:
            try:
                if poppler_path and os.path.exists(poppler_path):
                    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
                    successful_path = poppler_path
                    print(f"    ✅ Başarılı! Kullanılan yol: {poppler_path}")
                elif poppler_path is None:
                    images = convert_from_path(pdf_path, dpi=dpi)
                    successful_path = "PATH"
                    print(f"    ✅ Başarılı! PATH ortam değişkeninden")
                else:
                    continue
                    
                break
                
            except Exception as e:
                if poppler_path:
                    print(f"    ❌ {poppler_path} başarısız: {str(e)}")
                continue
        
        if not images:
            print(f"    ❌ PDF işlenemedi: {os.path.basename(pdf_path)}")
            return []
        
        # Geçici dosyalar oluştur
        image_paths = []
        for i, image in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                image_path = temp_file.name
            
            image.save(image_path, 'JPEG', quality=95)
            image_paths.append(image_path)
            print(f"      Sayfa {i+1}/{len(images)} işlendi")
        
        return image_paths
        
    except Exception as e:
        print(f"    ❌ PDF işleme hatası: {str(e)}")
        return []

def prepare_image(image_path, target_size=(800, 600)):
    """Görüntüyü hazırlar ve boyutlandırır"""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Oranı koruyarak yeniden boyutlandır
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Beyaz arka plan üzerine ortala
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        x = (target_size[0] - image.width) // 2
        y = (target_size[1] - image.height) // 2
        new_image.paste(image, (x, y))
        
        return new_image
    except Exception as e:
        print(f"    ❌ Görüntü hazırlama hatası: {str(e)}")
        return None

def clean_text(text):
    """Metin temizleme - OCR eğitimi için optimize edilmiş"""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:  # Boş satırları atla
            # Çok uzun satırları böl (OCR için maksimum 100 karakter)
            if len(line) > 100:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word + " ") <= 100:
                        current_line += word + " "
                    else:
                        if current_line:
                            cleaned_lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    cleaned_lines.append(current_line.strip())
            else:
                cleaned_lines.append(line)
    
    # En fazla 10 satır al (model eğitimi için)
    if len(cleaned_lines) > 10:
        cleaned_lines = cleaned_lines[:10]
    
    return '\n'.join(cleaned_lines)

def main():
    print("📚 OCR Veri Seti Hazırlama (PDF-TXT)")
    print("=" * 50)
    
    # Veri seti yapısını oluştur
    create_dataset_structure()
    
    # raw_data klasörünü kontrol et
    source_dir = "raw_data"
    
    if not os.path.exists(source_dir):
        print(f"❌ {source_dir} klasörü bulunamadı!")
        print("Lütfen raw_data klasörüne PDF ve TXT dosyalarınızı koyun.")
        return
    
    # Dosyaları listele
    files = os.listdir(source_dir)
    pdf_files = [f for f in files if f.endswith('.pdf')]
    txt_files = [f for f in files if f.endswith('.txt')]
    
    print(f"\n📁 {source_dir} klasörü içeriği:")
    print(f"  PDF dosyaları: {len(pdf_files)}")
    print(f"  TXT dosyaları: {len(txt_files)}")
    
    if len(pdf_files) == 0:
        print("\n❌ PDF dosyası bulunamadı!")
        print("Lütfen raw_data klasörüne PDF dosyalarınızı koyun.")
        return
    
    # PDF-TXT çiftlerini bul
    file_pairs = []
    for pdf_file in pdf_files:
        base_name = os.path.splitext(pdf_file)[0]
        txt_file = base_name + '.txt'
        
        if txt_file in txt_files:
            pdf_path = os.path.join(source_dir, pdf_file)
            txt_path = os.path.join(source_dir, txt_file)
            file_pairs.append((pdf_path, txt_path))
            print(f"  ✓ {pdf_file} + {txt_file}")
        else:
            print(f"  ❌ {pdf_file} için etiket dosyası bulunamadı: {txt_file}")
    
    if not file_pairs:
        print("\n❌ Eşleşen PDF-TXT çifti bulunamadı!")
        print("Her PDF dosyası için aynı isimde bir .txt dosyası olmalı.")
        print("Örnek: document.pdf + document.txt")
        return
    
    print(f"\n📊 Toplam {len(file_pairs)} PDF-TXT çifti bulundu.")
    
    # Dosyaları karıştır ve böl
    random.shuffle(file_pairs)
    
    total_files = len(file_pairs)
    train_size = int(total_files * 0.7)  # %70 eğitim
    val_size = int(total_files * 0.15)   # %15 doğrulama
    # Geri kalan %15 test
    
    train_files = file_pairs[:train_size]
    val_files = file_pairs[train_size:train_size + val_size]
    test_files = file_pairs[train_size + val_size:]
    
    print(f"\n📈 Veri seti bölümü:")
    print(f"  🎯 Eğitim: {len(train_files)} PDF")
    print(f"  🎯 Doğrulama: {len(val_files)} PDF")
    print(f"  🎯 Test: {len(test_files)} PDF")
    
    # Her set için dosyaları işle
    total_processed = 0
    
    for files, target_dir, set_name in [(train_files, 'train_data', 'Eğitim'), 
                                       (val_files, 'val_data', 'Doğrulama'),
                                       (test_files, 'test_images', 'Test')]:
        
        if not files:
            print(f"⚠️  {set_name} seti için dosya yok, atlanıyor...")
            continue
            
        print(f"\n🔄 {set_name} seti işleniyor ({target_dir})...")
        
        for i, (pdf_path, txt_path) in enumerate(files, 1):
            try:
                print(f"\n  [{i}/{len(files)}] İşleniyor: {os.path.basename(pdf_path)}")
                
                # Etiket dosyasını oku
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                # Metni temizle
                text_content = clean_text(text_content)
                
                if not text_content:
                    print(f"    ⚠️  Boş metin, atlanıyor...")
                    continue
                
                # PDF'yi görüntülere dönüştür
                image_paths = convert_pdf_to_images(pdf_path)
                
                if not image_paths:
                    print(f"    ❌ PDF işlenemedi, atlanıyor...")
                    continue
                
                # Her sayfa için görüntü ve etiket dosyası oluştur
                for j, img_path in enumerate(image_paths):
                    # Görüntüyü hazırla
                    image = prepare_image(img_path)
                    if image:
                        # Yeni dosya adları oluştur
                        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                        new_image_name = f"{base_name}_page_{j+1}.jpg"
                        new_label_name = f"{base_name}_page_{j+1}.txt"
                        
                        target_img_path = os.path.join(target_dir, new_image_name)
                        target_label_path = os.path.join(target_dir, new_label_name)
                        
                        # Görüntüyü kaydet
                        image.save(target_img_path, 'JPEG', quality=95)
                        
                        # Etiket dosyasını kaydet
                        with open(target_label_path, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                        
                        print(f"    ✅ Sayfa {j+1} kaydedildi: {new_image_name}")
                        total_processed += 1
                    
                    # Geçici dosyayı sil
                    try:
                        os.remove(img_path)
                    except:
                        pass
                
            except Exception as e:
                print(f"    ❌ Hata: {str(e)}")
                continue
    
    print(f"\n🎉 Veri seti hazırlama tamamlandı!")
    print(f"📊 Toplam {total_processed} görüntü-etiket çifti oluşturuldu.")
    
    # Özet bilgi
    print(f"\n📁 Oluşturulan dosyalar:")
    for target_dir in ['train_data', 'val_data', 'test_images']:
        if os.path.exists(target_dir):
            jpg_count = len([f for f in os.listdir(target_dir) if f.endswith('.jpg')])
            txt_count = len([f for f in os.listdir(target_dir) if f.endswith('.txt')])
            print(f"  📂 {target_dir}: {jpg_count} görüntü, {txt_count} etiket")
    
    # Doğrulama
    print(f"\n🔍 Veri seti doğrulaması:")
    all_valid = True
    
    for directory in ['train_data', 'val_data']:
        if os.path.exists(directory):
            jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
            txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
            
            if len(jpg_files) == len(txt_files) and len(jpg_files) > 0:
                print(f"  ✅ {directory}: {len(jpg_files)} çift - BAŞARILI")
            else:
                print(f"  ❌ {directory}: Dosya sayıları eşleşmiyor!")
                all_valid = False
        else:
            print(f"  ❌ {directory}: Klasör bulunamadı!")
            all_valid = False
    
    if all_valid and total_processed > 0:
        print(f"\n🚀 Veri seti hazır! Şimdi model eğitimini başlatabilirsiniz:")
        print(f"   python train_model.py")
    else:
        print(f"\n⚠️  Veri setinde sorunlar var. Lütfen tekrar kontrol edin.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {str(e)}")
        print("Lütfen hata mesajını kontrol edin.")