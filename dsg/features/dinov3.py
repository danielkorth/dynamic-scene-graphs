from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch


class DINOv3:
    def __init__(self, model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"):
        """
        Initialize DINOv3 model for feature extraction.

        Args:
            model_name (str): Name of the DINOv3 model to load.
                              Examples:
                              - "facebook/dinov3-vits16-pretrain-lvd1689m"
                              - "facebook/dinov3-vitsplus-pretrain-lvd1689m"
                              - "facebook/dinov3-vit7b16-pretrain-lvd1689m"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_features(self, image_path: str) -> torch.Tensor:
        """
        Extract a global embedding from an image.

        Prefers the model's pooled output when available; otherwise falls back
        to the CLS token from the last hidden state.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Global embedding tensor of shape [1, hidden_size].
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Prefer pooled output when available (as per HF DINOv3 docs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        # Fallback to CLS embedding from last hidden state
        last_hidden_states = (
            outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        )
        cls_embedding = last_hidden_states[:, 0, :]
        return cls_embedding
