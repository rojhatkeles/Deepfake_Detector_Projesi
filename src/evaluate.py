import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
from dataset import FaceDeepfakeDataset
from model import get_deepfake_detector_model

def evaluate_test_set():
    print("🚀 Deepfake Modeli (Otonom Laboratuvar Merkezi) Başlatılıyor...\n")
    
    # 1. Klasör Yolları
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(project_root, 'archive-2', 'real_vs_fake', 'real-vs-fake', 'test')
    model_path = os.path.join(project_root, 'saved_models', 'best_model.pth')
    
    if not os.path.exists(test_dir):
        print(f"HATA: Test klasörü bulunamadı: {test_dir}")
        return
        
    if not os.path.exists(model_path):
        print(f"HATA: Eğitilmiş model bulunamadı: {model_path}")
        return

    # 2. Cihaz Hazırlığı
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🖥️  Donanım İvmelendirmesi: {device}")
    
    # 3. Modeli Yükleme
    model = get_deepfake_detector_model(freeze_features=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 4. Veri İşleme (Kaggle Test Seti zaten kesik/kırpık olduğu için doğrudan alıyoruz)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FaceDeepfakeDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"📂 Toplam test edilecek fotoğraf sayısı: {len(test_dataset)}")
    
    # İstatistikler (Confusion Matrix Temelleri)
    true_positives = 0  # Başarılı (Sahteyi bildi)
    true_negatives = 0  # Başarılı (Gerçeği bildi)
    false_positives = 0 # Alarm Hatası (Gerçeğe Sahte Dedi)
    false_negatives = 0 # Kandırıldı (Sahteye Gerçek Dedi)
    
    start_time = time.time()
    print("\n⏳ Yapay zeka binlerce fotoğrafı süzgeçten geçiriyor... Lütfen bekleyin.\n")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Matris Testi", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            # Arayüzdeki (app.py) agresif sınırımızı burada referans alıyoruz!
            preds = (probs > 0.40).float()
            
            for i in range(len(labels)):
                p = preds[i].item()
                t = labels[i].item()
                
                if t == 1.0 and p == 1.0: true_positives += 1
                elif t == 0.0 and p == 0.0: true_negatives += 1
                elif t == 0.0 and p == 1.0: false_positives += 1
                elif t == 1.0 and p == 0.0: false_negatives += 1

    total_time = time.time() - start_time
    total = len(test_dataset)
    accuracy = (true_positives + true_negatives) / total * 100
    
    print("\n" + "="*70)
    print("📊 BİLİMSEL TEST SONUÇLARI (Confusion Matrix)")
    print("="*70)
    print(f"Genel Doğruluk (Accuracy)  : % {accuracy:.2f}")
    print(f"Test Edilen Fotoğraf       : {total} adet")
    print(f"Toplam Test Süresi         : {total_time:.1f} saniye")
    print("\n--- DETAYLI SKOR TABLOSU ---")
    print(f"✔️ TAM İSABET - Gerçek yüzleri tanıdı        (True Negative)  : {true_negatives}")
    print(f"✔️ TAM İSABET - Sahte yüzleri başarıyla avladı (True Positive)  : {true_positives}")
    print(f"❌ HATA (Paranoya) - Gerçek bir yüze 'Sahte' dedi (False Positive): {false_positives}")
    print(f"❌ ÖLÜMCÜL HATA - Sahte yüze kanıp 'Gerçek' dedi (False Negative) : {false_negatives}")
    print("="*70)

if __name__ == "__main__":
    evaluate_test_set()
