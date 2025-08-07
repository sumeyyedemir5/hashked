import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..')) 

TOP_LEVEL_COMMON_PARENT = os.path.abspath(os.path.join(PROJECT_ROOT, '..')) # Bu kısım doğru kalıyor
if TOP_LEVEL_COMMON_PARENT not in sys.path:
    sys.path.append(TOP_LEVEL_COMMON_PARENT)

from U2NET.data_loader import RescaleT, ToTensor
from U2NET.model.u2net import U2NET

UPLOADS_DIR = os.path.join(PROJECT_ROOT, 'uploads')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'processed')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'u2net.pth')
INPUT_IMAGE = os.path.join(UPLOADS_DIR, 'input_2.png')
OUTPUT_IMAGE = os.path.join(PROCESSED_DIR, 'input_clean2.png')

def load_model(model_path):
    print("📦 Model yükleniyor...")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([RescaleT(320), ToTensor()])
    sample = {'image': image, 'label': image, 'imidx': 0}
    sample = transform(sample)
    image_tensor = sample['image'].unsqueeze(0)
    return image, image_tensor

def apply_mask(original, mask):
    original = original.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
    original = np.array(original).astype(np.uint8)

    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = (mask * 255).astype(np.uint8)

    rgba_image = np.dstack((original, mask))
    return Image.fromarray(rgba_image, 'RGBA')

def segment_image():
    print(f"DEBUG: Yükleme Dizini (UPLOADS_DIR): {UPLOADS_DIR}")
    print(f"DEBUG: Girdi Görseli (INPUT_IMAGE): {INPUT_IMAGE}")
    print(f"DEBUG: Model Yolu (MODEL_PATH): {MODEL_PATH}")

    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Girdi görseli bulunamadı: {INPUT_IMAGE}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
        return

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        print(f"Oluşturulan dizin: {PROCESSED_DIR}")

    try:
        net = load_model(MODEL_PATH)
        original, image_tensor = preprocess_image(INPUT_IMAGE)
        image_tensor = Variable(image_tensor)

        with torch.no_grad():
            d1, *_ = net(image_tensor)
            mask = d1[0][0].cpu().data.numpy()

        result = apply_mask(original, mask)
        result.save(OUTPUT_IMAGE)
        print(f"✅ Segmentasyon tamamlandı. Çıktı: {OUTPUT_IMAGE}")

    except Exception as e:
        print(f"🛑 Bir hata oluştu: {e}")


if __name__ == "__main__":
    segment_image()