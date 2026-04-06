import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
import shutil

def crop_faces(source_dir, target_dir):
    # MTCNN kütüphanesinin Apple MPS üzerinde NMS (Non-Maximum Suppression) matris 
    # hataları vererek çöktüğünü tespit ettik. İşlemler hata verip yutulduğu için 0 yüz bulundu.
    # Bu nedenle bu islem icin stabil olan CPU donanımını tetikliyoruz.
    device = torch.device('cpu')
    
    # image_size: kırpılan yüz resminin boyutu
    # margin: yüzden ne kadar dışarı taşılacağı. Modelin çene ve yüz
    # sınırını görmesi için biraz pay bırakmak (orn: 20-40) iyidir.
    mtcnn = MTCNN(image_size=224, margin=30, keep_all=False, post_process=False, device=device)
    
    print(f"\n[AI-Bot] MTCNN Yüz Tespiti Taraması Başlıyor...\nKaynak: {source_dir}\nHedef: {target_dir}\n")
    
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            in_dir = os.path.join(source_dir, split, label)
            out_dir = os.path.join(target_dir, split, label)
            os.makedirs(out_dir, exist_ok=True)
            
            if not os.path.exists(in_dir):
                continue
                
            images = [f for f in os.listdir(in_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            # Veri kaybetmemek için sayaçlar
            cropped_count = 0
            original_count = 0
            
            for img_name in tqdm(images, desc=f"{split}/{label} (Yüz Taraması)"):
                img_path = os.path.join(in_dir, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    # Yüzü Algıla ve klasöre kırpılmış haliyle kaydet
                    # Eger yuz bulursa MTCNN out_dir icine direkt resmi kaydedecek (save_path sayesinde).
                    # save_path parametresi tensor donusu yerine direk kayit islemi yaptirip None dondurebilir, ama save isini kendimiz de yapabiliriz.
                    
                    face_tensor = mtcnn(img, save_path=os.path.join(out_dir, img_name))
                    
                    if face_tensor is not None:
                        cropped_count += 1
                    else:
                        # Eğer yapay zeka yüz bulamazsa (arka planı çok bozuksa)
                        # Hata almamak için orjinal resmin kendisini kopyalayalım
                        shutil.copy(img_path, os.path.join(out_dir, img_name))
                        original_count += 1
                except Exception as e:
                    # Hatalar önceden pas geçildiği için sorunu göremiyorduk.
                    print(f"\nHATA ({img_name}): {e}")
                    
            print(f" -> {split}/{label}: {cropped_count} yüz başarıyla bulundu. {original_count} resimde yüz tespit edilemedi, orjinalleri kopyalandı.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = os.path.join(project_root, 'data')
    target = os.path.join(project_root, 'data_cropped')
    
    crop_faces(source, target)
    print("\n[AI-Bot] Mükemmel! Veri setindeki gereksiz arka planlar temizlendi.")
    print("Artık yeni eğitimimiz 'data_cropped' klasörü üzerinden gerçekleşmeli.")
