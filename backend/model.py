"""Loads the fine-tuned real/fake face classifier and exposes inference + Grad-CAM."""
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "classifier.pt"


class DeepfakeClassifier:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No trained checkpoint at {checkpoint_path}. "
                "Run training/dataset_prep.py and training/train.py first."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.classes = checkpoint["classes"]  # ["fake", "real"], sigmoid output = P(real)
        self.image_size = checkpoint["image_size"]
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

        self.model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        # conv_head is EfficientNet-B0's last conv block before pooling - a reasonable
        # default target layer for Grad-CAM on this architecture.
        self.cam = GradCAM(model=self.model, target_layers=[self.model.conv_head])

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor).squeeze()
            prob_real = torch.sigmoid(logit).item()
        prob_fake = 1.0 - prob_real
        label = "real" if prob_real >= 0.5 else "fake"
        return {
            "label": label,
            "confidence": max(prob_real, prob_fake),
            "prob_real": prob_real,
            "prob_fake": prob_fake,
        }

    def gradcam_heatmap(self, image: Image.Image) -> Image.Image:
        """Returns an RGB overlay showing which regions most influenced the model's output.
        This explains the model's attention, not a verified "artifact location" - see README."""
        rgb = image.convert("RGB").resize((self.image_size, self.image_size))
        rgb_float = np.array(rgb).astype(np.float32) / 255.0
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        grayscale_cam = self.cam(input_tensor=tensor)[0]
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        return Image.fromarray(overlay)
