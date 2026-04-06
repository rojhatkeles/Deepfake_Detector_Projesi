import torch
import torch.nn as nn
from torchvision import models

def get_deepfake_detector_model(freeze_features=True):
    """
    ImageNet'te ön eğitimli VGG16 modelini yükler ve 
    ikili sınıflandırma (Binary Classification) yapacak şekilde yapısını değiştirir.
    """
    # 1. Weights: Ön eğitimli ağırlıkları internetten (veya önbellekten) indir
    # VGG16_Weights.IMAGENET1K_V1, modelin ImageNet'teki nesneleri tanırken öğrendiği genel
    # çizgi, köşe ve renk geçişlerini içerir.
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # 2. İLERİ SEVİYE FINE-TUNING (Deep Learning Optimizasyonu)
    if freeze_features:
        # Önce VGG16'nın tüm özellik çıkarıcılarını (features) dondur
        for param in model.features.parameters():
            param.requires_grad = False
            
        # VGG16'nın SADECE son Evrişimsel (Convolutional) bloklarının buzlarını açıyoruz! (Katman 24'ten sonrası)
        # Neden? Çünkü ağın giriş kısımları (kenar/çizgi) değişmemeli ama 
        # en derindeki kısımlar spesifik Deepfake piksellerini öğrenmek zorundadır.
        for param in model.features[24:].parameters():
            param.requires_grad = True
            
    # 3. Modelin Karar (Sınıflandırma) Kısmını Güçlendirmek (Dropout & Extra Depth)
    num_features = model.classifier[6].in_features
    
    # Overfitting'i (Ezberlemeyi) kırıp atmak için basit bir Linear katman yerine 
    # Sequential (Zincirleme) ve Dropout barındıran güçlü bir zıh yaratıyoruz.
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5), # Her iterasyonda nöronların yarısını rastgele kapat (Ezberi bozar)
        nn.Linear(512, 1)
    )
    
    return model

# Test için (sadece bu dosya çalıştırıldığında aktif olur)
if __name__ == "__main__":
    # Apple Silicon (M1/M2/M3) kontrolü!
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")
    
    # Modeli oluştur ve seçilen cihaza yolla
    model = get_deepfake_detector_model(freeze_features=True)
    model = model.to(device)
    
    # Varsayımsal bir girdi (4 adet, 3 renk kanallı, 224x224 resim)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Çıktı boyutu (Batch Size, Çıktı Nöronu): {output.shape}") 
    # Beklenen çıktı boyutu: torch.Size([4, 1])
