import logging
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class TokenEmbedder:
    def __init__(self, model_path=None, device='cpu'):
        # Chuẩn hóa device string: '0' hoặc 0 → 'cuda:0', 'cpu' → 'cpu'
        if isinstance(device, str) and device.isdigit():
            device = f'cuda:{device}'
        elif isinstance(device, int):
            device = f'cuda:{device}'
        self.device = str(device)
        logger.info(f"Loading MobileNetV3-Small embedder on {self.device}")
        
        try:
            # Use MobileNetV3 Small as feature extractor
            self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            # Remove the classifier head, keep only features
            self.model.classifier = torch.nn.Identity()
            self.model.eval()
            self.model.to(self.device)
            
            if model_path:
                logger.info(f"Loading custom weights from {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info("Embedder loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            raise

    def extract(self, image):
        """Extract feature embedding for a single BGR image."""
        if image is None or image.size == 0:
            return None
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
            
        # Return as normalized numpy array
        embedding = features.cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def extract_batch(self, images):
        """Extract feature embeddings for a list of BGR images."""
        if not images:
            return []
            
        tensors = []
        for img in images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(img_rgb))
            
        batch_tensor = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch_tensor)
            
        embeddings = features.cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms!=0)
        
        return embeddings
