import os
import sys
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, UTILS_DIR)

from pix2vox_a import Pix2VoxA
from voxel import save_binvox

def remove_module_prefix(state_dict, prefix):
    """
    Prefix'i key'lere ekle (tam tersi davranış).
    Bu fonksiyon, state_dict içindeki key'leri,
    örneğin "vgg.0.weight" → "encoder.vgg.0.weight" gibi dönüştürür.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f"{prefix}.{key}"
        new_state_dict[new_key] = value
    return new_state_dict

# Görsel yolu (önceden temizlenmiş, işlenmiş)
image_path = os.path.join(PROJECT_ROOT, "processed", "input_clean2.png")
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Pix2Vox giriş boyutu
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)  # batch dim [1,3,32,32]

from types import SimpleNamespace

cfg = SimpleNamespace(
    NETWORK=SimpleNamespace(
        TCONV_USE_BIAS=True,
        LEAKY_VALUE=0.2  # veya kodda beklenen değeri
    )
)

model = Pix2VoxA(cfg)



# Checkpoint yolu (önceden indirilmiş model ağırlıkları)
checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "pix2vox-a.pth")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)


# Model ağırlıklarını yükle
model.load_state_dict(remove_module_prefix(checkpoint["encoder_state_dict"], "encoder"), strict=False)
model.load_state_dict(remove_module_prefix(checkpoint["decoder_state_dict"], "decoder"), strict=False)
model.load_state_dict(remove_module_prefix(checkpoint["merger_state_dict"], "merger"), strict=False)

model.eval()

# Tahmin yap
with torch.no_grad():
    voxel = model(image_tensor).squeeze().numpy()

# Çıktı dosyası yolu
output_path = os.path.join(PROJECT_ROOT, "outputs", "result.binvox")

# Klasör yoksa oluştur
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Sonucu kaydet
save_binvox(voxel > 0.3, output_path)

print(f"3D voxel modeli oluşturuldu: {output_path}")
