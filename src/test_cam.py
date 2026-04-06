import torch
import os
from PIL import Image
import numpy as np

from model import get_deepfake_detector_model
from explainer import get_gradcam_heatmap
from torchvision import transforms

def main():
    print("\n[++] XAI (Explainable AI) Grad-CAM Dedektörü Başlatılıyor...\n")
    
    # 1. Modeli ve Cihazı Ayarla
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Grad-CAM'in matematiksel olarak ısı haritasını çıkarabilmesi için 
    # özellik katmanlarının türevlerine (Gradients) ihtiyacı var. 
    # Eğitim bittiği için artık dondurmaya gerek yok, False verip kilitleri açıyoruz.
    model = get_deepfake_detector_model(freeze_features=False)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(project_root, 'saved_models', 'best_model.pth')
    
    if not os.path.exists(weights_path):
        print(f"HATA: Eğitilmiş ağırlık dosyası bulunamadı! Yol: {weights_path}")
        return
        
    # PyTorch State Dict (Ağırlıkları) Yükleme
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval() # Geri yayılımı ve dropout'u kapatır, Salt Tahmin (Inference) Modu
    print("--> 1. Epoch'ta kaydedilen en zeki ağ ağırlıkları başarıyla yüklendi!\n")

    # 2. Rastgele Bir "Sahte" (Fake) Görsel Seçelim
    val_fake_dir = os.path.join(project_root, 'data', 'val', 'fake')
    if not os.path.exists(val_fake_dir):
         print("Test edilecek val/fake klasörü bulunamadı.")
         return
         
    sample_images = os.listdir(val_fake_dir)
    img_path = os.path.join(val_fake_dir, sample_images[0]) # Dizindeki ilk sahte fotoğrafı alıyoruz
    print(f"Modelin İnceleyeceği Görsel: {sample_images[0]}")
    
    original_pil = Image.open(img_path).convert('RGB')
    
    # 3. Tensör Dönüşümleri (Modelin sadece sayısal matrisle çalıştığını hatırla)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Batch boyutu için .unsqueeze(0) ile [1, 3, 224, 224] shape'ini elde ediyoruz
    input_tensor = transform(original_pil).unsqueeze(0).to(device)
    
    # 4. Modeli Sorgulama (Inference)
    with torch.no_grad():
         output = model(input_tensor)
         # Çıktı [1, 1] logit değeridir. Sigmoid ile 0-1 arası (Yüzdelik) ihtimaline sıkıştırıyoruz
         prob = torch.sigmoid(output).item()
         prediction_text = "SAHTE (FAKE)!" if prob > 0.5 else "GERÇEK (REAL)"
         print(f"Modelin Tahmini: {prediction_text} (Sahtelik Olasılığı: %{prob*100:.2f})\n")
         
    # 5. Kara Kutuyu Açıklamak: Grad-CAM Isı Haritasını Üret!
    print("Isı Haritası dökümleri (Gradient Maps) oluşturuluyor...")
    heatmap_img = get_gradcam_heatmap(model, input_tensor, original_pil)
    
    # 6. Görüntüyü Ana Klasöre Kaydet
    output_path = os.path.join(project_root, "gradcam_output_demo.jpg")
    heatmap_img.save(output_path)
    print(f"\n[!!!] ŞOV ZAMANI! Isı haritası içeren kanıtlı fotoğraf kaydedildi.")
    print(f"Lütfen soldaki dizinden şu dosyayı açıp incele: gradcam_output_demo.jpg")

if __name__ == "__main__":
    main()
