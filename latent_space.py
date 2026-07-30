import os
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.vae import VAE

# ==========================================
# Configuration
# ==========================================

IMAGE_SIZE = 128
LATENT_DIM = 128
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "checkpoints/best_vae.pth"

DATASET_PATH = "dataset/bottle/test"

OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Dataset
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Images Loaded : {len(dataset)}")

# ==========================================
# Load Model
# ==========================================

model = VAE(latent_dim=LATENT_DIM).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model Loaded Successfully")

# ==========================================
# Extract Latent Vectors
# ==========================================

latent_vectors = []

with torch.no_grad():

    for images, _ in loader:

        images = images.to(DEVICE)

        _, mu, _ = model(images)

        latent_vectors.append(mu.cpu())

latent_vectors = torch.cat(latent_vectors, dim=0)

print("Latent Shape :", latent_vectors.shape)

# ==========================================
# PCA
# ==========================================

pca = PCA(n_components=2)

latent_2d = pca.fit_transform(
    latent_vectors.numpy()
)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(8,8))

plt.scatter(
    latent_2d[:,0],
    latent_2d[:,1],
    s=35,
    alpha=0.8
)

plt.title("VAE Latent Space (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "latent_space.png"
    )
)

plt.show()

print("Latent space saved successfully!")