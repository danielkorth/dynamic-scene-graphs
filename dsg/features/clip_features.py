from PIL import Image
import torch
from typing import List, Union
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
)


class CLIPFeatures:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP vision encoder for feature extraction.

        Args:
            model_name (str): Hugging Face model id for the CLIP vision encoder.
                              Defaults to "openai/clip-vit-base-patch32".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor and CLIP vision model (with projection head)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer and CLIP text model (with projection head)
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_name)

        # Move text model to device
        self.text_model = self.text_model.to(self.device)
        self.text_model.eval()

    def extract_vision_features(self, image_path: str) -> torch.Tensor:
        """
        Extract projected CLIP vision embeddings from an image path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Projected image embeddings of shape [1, projection_dim].
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if not hasattr(outputs, "image_embeds") or outputs.image_embeds is None:
            raise RuntimeError("CLIPVisionModelWithProjection did not return image_embeds")
        return outputs.image_embeds

    def extract_text_features(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract projected CLIP text embeddings for one or more texts.

        Args:
            texts (str | list[str]): Single text or list of texts.

        Returns:
            torch.Tensor: Projected text embeddings of shape [batch_size, projection_dim].
        """
        if isinstance(texts, str):
            texts = [texts]
        enc = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {key: value.to(self.device) for key, value in enc.items()}
        with torch.no_grad():
            outputs = self.text_model(**enc)
        if not hasattr(outputs, "text_embeds") or outputs.text_embeds is None:
            raise RuntimeError("CLIPTextModelWithProjection did not return text_embeds")
        return outputs.text_embeds


# Example usage
if __name__ == "__main__":
    extractor = CLIPFeatures()
    image_path = "/local/home/dkorth/Projects/dynamic-scene-graphs/outputs/2025-08-02/09-13-00/crop/cropped_image_0.jpg"
    img_emb = extractor.extract_vision_features(image_path)
    txt_emb = extractor.extract_text_features(["a photo of a cat", "a photo of a dog"])
    print("CLIP projected image embedding:", img_emb.shape)
    print("CLIP projected text embeddings:", txt_emb.shape)
