import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDeepfakeDataset(Dataset):
    """
    Kişiselleştirilmiş PyTorch Dataset Sınıfı.
    Görselleri diskten okur, tensor yapılarına çevirir ve etiketleri ile döndürür.
    [0] -> Gerçek (Real)
    [1] -> Sahte (Fake)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Klasör yapısının data_dir içerisinde 'real' ve 'fake' alt klasörleri
        # olacak şekilde tasarlandığını varsayıyoruz.
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')
        
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(real_dir, img_name))
                    self.labels.append(0.0) # Binary cross entropy için float
                    
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1.0)
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # MPS Cihaz kısıtlaması burada Dataset düzeyinde YAPILMAZ!
        # Dataloader veriyi hazırlarken CPU'yu kullanır, donanım ivmelendirmesi (device='mps')
        # eğitim döngüsünde (train.py içinde) modele ve tensorlere .to(device) olarak verilir.
        return image, torch.tensor(label, dtype=torch.float32)

def get_dataloaders(data_dir, batch_size=32):
    """
    Verilen kök dizindeki train, val (ve varsa test) klasörleri için DataLoader'lar oluşturur.
    """
    # 5 GÜNLÜK GELİŞTİRME PLANI: X-Augmentation (Blur & Jitter)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    train_dataset = FaceDeepfakeDataset(train_dir, transform=train_transform)
    val_dataset = FaceDeepfakeDataset(val_dir, transform=val_transform)
    test_dataset = FaceDeepfakeDataset(test_dir, transform=val_transform) if os.path.exists(test_dir) else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if test_dataset else None
    
    return train_loader, val_loader, test_loader
