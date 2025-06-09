import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load model ResNet50
model = models.resnet50(pretrained=True)

# Load bobot dari file lokal
model.load_state_dict(torch.load('resnet50.pth'))
model.eval()  # mode evaluasi

# Hilangkan bagian klasifikasi akhir (ambil fitur saja)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Resize and preprocessing for ResNet50 
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize 224x224
    transforms.ToTensor(),  # Ubah format ke tensor pytorch
    transforms.Normalize(  # sesuai mean/std ImageNet # Normalisasi channel warna
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load gambar dan ubah ke tensor
img = Image.open("original_20250419_213428.jpg").convert("RGB")

input_tensor = transform(img).unsqueeze(0)  # tambah batch dimensi

# Ekstrak fitur
with torch.no_grad():
    features = feature_extractor(input_tensor)

print("Fitur shape:", features.shape)  # biasanya [1, 2048, 1, 1]
print("Fitur :", features)  # biasanya [1, 2048, 1, 1]
