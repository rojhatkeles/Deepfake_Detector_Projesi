import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_deepfake_detector_model
import os

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.00001):
    # 1. Cihaz Ataması (Donanım Kısıtlaması, MPS Entegrasyonu)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{device}] üzerinde eğitime başlanıyor...")

    # 2. Dataset ve Dataloader nesnelerinin getirilmesi
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)
    
    # 3. Modelin yaratılması ve MPS'e atılması
    model = get_deepfake_detector_model(freeze_features=True)
    model = model.to(device)
    
    # 4. Kayıp (Loss) Fonksiyonu: İkili Çapraz Entropi (Binary Cross Entropy)
    criterion = nn.BCEWithLogitsLoss()
    
    # 5. Optimizatör: Sadece gradient isteyen (requires_grad=True) parametreleri
    # optimize edeceği için Ram ve CPU performansına çok iyi gelecek.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    # Eğitim duraklarsa hızı otomatik küçültecek yapay zeka şanzımanı
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
    # Klasör yolları ayarları
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(num_epochs):
        model.train() # Ağı eğitim katmanlarına hazırla (Dropout gibi fonksiyonlar aktifleşir)
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in train_loader:
            # Kritik Nokta: Tensörler mutlaka donanıma (mps) alınmalıdır. (Aksinde cras islenir)
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device) # Etiketler [1,0,1,0] formatından matrikse dökülmeli.
            
            # 1. İleri Yayılım (Forward Pass)
            optimizer.zero_grad() # Her döngüde bir önceki adımın gradyan türevlerini temizlemeliyiz.
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            
            # 2. Geri Yayılım (Backpropagation) ve Ağırlık Güncellemesi
            loss.backward()  # Hata payını modeldeki ağırlıklara Zincir Kuralıyla (Chain Rule) geri orantıla
            optimizer.step() # Yeni gradyanlara göre ağırlıkları azalt/arttır 
            
            # İstatistik Hesaplama
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
            
        epoch_loss = running_loss / total_preds
        epoch_acc = correct_preds / total_preds
        
        # --- DOĞRULAMA (VALIDATION) DÖNGÜSÜ ---
        # Amaç modelin Train'i ezberlemediğinden (Overfitting yapmadığından) emin olmaktır.
        model.eval() 
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # Geri Yayılım kapatılır, ezberleme performansı büyük ölçüde artar.
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.view(-1, 1).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        # Loss oranını şanzımana (Scheduler) gönder ki gaza mı basacak frene mi karar versin
        scheduler.step(val_epoch_loss)
        
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
              f"Train Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} - Acc: {val_epoch_acc:.4f}")
              
        # Model Checkpoint: Her epokta daha da akıllanan modeli kaydet.
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"--> Validation Loss azaldı, yeni Checkpoint oluşturuldu: {model_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 140 BINLIK DEVASA EĞİTİM MODU: Makineyi sabaha kadar tam güç çalıştıracak veri yolu!
    data_dir = os.path.join(project_root, 'archive-2', 'real_vs_fake', 'real-vs-fake')
    
    if os.path.exists(os.path.join(data_dir, 'train')):
        print("🔥 --- 140.000 FOTOĞRAFLIK DEVASA EĞİTİM MOTORU BAŞLATILIYOR --- 🔥")
        print(f"Hedef Veri Seti Okundu: {data_dir}")
        train_model(data_dir=data_dir, num_epochs=10)
    else:
        print(f"HATA: Dev arşiv bulunamadı. Beklenen yol: {data_dir}")
