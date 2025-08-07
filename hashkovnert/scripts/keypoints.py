import cv2
import mediapipe as mp
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
IMAGE_PATH = os.path.join(PROJECT_ROOT, 'uploads', 'input_2.png')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if image is None:
    print(f"❌ Görsel yüklenemedi. Yolu kontrol et: {IMAGE_PATH}")
    exit()

print(f"✅ Görsel yüklendi: {IMAGE_PATH}, boyut: {image.shape}")

results = pose.process(image_rgb)

if results.pose_landmarks:
    print("✅ Anahtar noktalar tespit edildi.")
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS
    )
    output_path = os.path.join(PROJECT_ROOT, 'processed', 'keypoints.png')
    cv2.imwrite(output_path, annotated_image)
    print(f"💾 Anahtar noktalı görsel kaydedildi: {output_path}")
else:
    print("❌ Anahtar noktalar tespit edilemedi.")
