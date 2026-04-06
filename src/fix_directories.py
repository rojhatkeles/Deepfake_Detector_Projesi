import os
import shutil
import random

def fix_directories():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    print("Klasör Mimarisi ve Veri Sızıntısı (Data Leakage) onarılıyor...")
    
    # 1. Hatalı alt klasör taşıma (training_real -> real içindeki dosyaları direk real'a çekiyoruz)
    for label, bad_folder in [('real', 'training_real'), ('fake', 'training_fake')]:
        train_bad = os.path.join(base_dir, 'train', label, bad_folder)
        train_good = os.path.join(base_dir, 'train', label)
        
        if os.path.exists(train_bad):
            for file in os.listdir(train_bad):
                shutil.move(os.path.join(train_bad, file), os.path.join(train_good, file))
            shutil.rmtree(train_bad)
            
    # 2. Validation klasöründeki kopyaları SİLME (Data Leakage tehlikesi)
    for label, bad_folder in [('real', 'training_real'), ('fake', 'training_fake')]:
        val_bad = os.path.join(base_dir, 'val', label, bad_folder)
        val_good = os.path.join(base_dir, 'val', label)
        
        if os.path.exists(val_bad):
            shutil.rmtree(val_bad)
            
        # Daha doğrusu val içini tamamen temizleyelim (sıfırdan %20 atacağız)
        if os.path.exists(val_good):
            for f in os.listdir(val_good):
                os.remove(os.path.join(val_good, f))
                
    # 3. Eğitimdeki görsellerin %20'sini rastgele seçip Validation'a aktarma
    random.seed(42)
    for label in ['real', 'fake']:
        train_folder = os.path.join(base_dir, 'train', label)
        val_folder = os.path.join(base_dir, 'val', label)
        os.makedirs(val_folder, exist_ok=True)
        
        if os.path.exists(train_folder):
            images = [f for f in os.listdir(train_folder) if f.lower().endswith(('.jpg', '.png'))]
            random.shuffle(images)
            
            val_count = int(len(images) * 0.2)
            val_images = images[:val_count]
            
            for img in val_images:
                shutil.move(os.path.join(train_folder, img), os.path.join(val_folder, img))
            
    print("Düzeltme Tamamlandı! Train ve Val kümeleri bir AI sisteminin beklediği standartta onarıldı.")

if __name__ == "__main__":
    fix_directories()
