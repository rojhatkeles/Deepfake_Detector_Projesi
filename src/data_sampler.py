import os
import random
import shutil
import sys
from tqdm import tqdm

def sample_dataset(source_path, target_path, total_per_class=5000):
    print(f"\n[AI-Bot] Arşiv Taranıyor: {source_path}")
    
    all_real = []
    all_fake = []
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(root, file)
                path_lower = full_path.replace('\\', '/').lower()
                
                if '/fake' in path_lower or '/synthetic' in path_lower:
                    all_fake.append(full_path)
                elif '/real' in path_lower or '/original' in path_lower:
                    all_real.append(full_path)
                    
    print(f"Toplam Bulunan -> Real: {len(all_real)}, Fake: {len(all_fake)}")
    
    if len(all_real) < total_per_class or len(all_fake) < total_per_class:
        print("HATA: İstediğiniz sayıda resim yok.")
        return
        
    random.seed(42) 
    random.shuffle(all_real)
    random.shuffle(all_fake)
    
    sel_real = all_real[:total_per_class]
    sel_fake = all_fake[:total_per_class]
    
    train_real, val_real = sel_real[:4000], sel_real[4000:]
    train_fake, val_fake = sel_fake[:4000], sel_fake[4000:]
    
    tasks = [
        (train_real, os.path.join(target_path, 'train', 'real')),
        (val_real, os.path.join(target_path, 'val', 'real')),
        (train_fake, os.path.join(target_path, 'train', 'fake')),
        (val_fake, os.path.join(target_path, 'val', 'fake')),
    ]
    
    if os.path.exists(target_path):
        shutil.rmtree(target_path)  # Eski zayıf modeli sil
        
    print("\n[AI-Bot] 10.000 Şanslı Yüz Kopyalanıyor...\n")
    for file_list, dest_dir in tasks:
        os.makedirs(dest_dir, exist_ok=True)
        for img_path in tqdm(file_list, desc=f"Transfer: {os.path.basename(os.path.dirname(dest_dir))}"):
            shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
            
    print(f"\n[DONE] 10.000 veri projene dizildi! Eğitime hazırsınız.")

if __name__ == '__main__':
    # Terminale yazmana gerek kalmadı, yolları kodun kalbine sabitliyoruz!
    source_folder = os.path.expanduser("~/Desktop/archive-2")
        
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    target_folder = os.path.join(project_root, 'data_cropped')
    
    print("Motor çalıştı, Masaüstündeki Arşive Gidiliyor...")
    sample_dataset(source_folder, target_folder)
