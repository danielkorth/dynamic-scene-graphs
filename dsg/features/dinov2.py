import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch

class DINOv2:
    def __init__(self, model_name="facebook/dinov2-base"):
        """
        Initialize DINOv2 model for feature extraction.
        
        Args:
            model_name (str): Name of the DINOv2 model to load
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, image_path):
        """
        Extract CLS embedding from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: CLS embedding tensor of shape [1, hidden_size]
        """
        # Load image
        image = Image.open(image_path)
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_states = outputs[0]  # shape: [batch_size, num_patches+1, hidden_size]
        
        # Extract CLS embedding
        cls_embedding = last_hidden_states[:, 0, :]  # [CLS] token: global image embedding
        
        return cls_embedding
