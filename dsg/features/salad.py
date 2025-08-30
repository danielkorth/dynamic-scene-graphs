import torch
import torchvision.transforms as T
from PIL import Image
import math

class SALAD:
    def __init__(self):
        """
        Initialize SALAD model for feature extraction.
        
        SALAD Constraint Strategy:
        ========================
        SALAD requires two constraints to be satisfied:
        1. Both dimensions must be divisible by 14 (for patch processing)
        2. (H×W)/(14×14) > 64, meaning H×W > 12544 (sufficient patches)
        
        Strategy:
        1. First attempt: Round to nearest multiples of 14
        2. Check if area constraint is satisfied (area > 12544)
        3. If not, scale up while preserving aspect ratio:
           - Calculate minimum dimensions using mathematical approach
           - For aspect ratio r = w/h:
             * If r ≥ 1: h = √(min_area/r), w = h×r
             * If r < 1: w = √(min_area×r), h = w/r
        4. Round up to next multiples of 14
        5. Verify both constraints are satisfied
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SALAD model
        print("Loading SALAD model...")
        self.model = torch.hub.load("serizba/salad", "dinov2_salad")
        self.model.eval()
        
        # Move model to device
        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model = self.model.to(self.device)
        
        # Define transforms
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        
        # SALAD constraint: (H*W)/(14*14) > 64, so H*W > 12544
        self.MIN_AREA = 64 * 14 * 14 + 1  # 12545 to ensure > constraint
    
    def _get_valid_salad_size(self, width, height):
        """
        Get image size that satisfies SALAD constraints:
        1. Both dimensions divisible by 14 (for patch processing)
        2. (H×W)/(14×14) > 64, i.e., H×W > 12544 (sufficient patches for SALAD)
        
        Strategy Implementation:
        - First attempt: Round to nearest multiples of 14
        - If area constraint not satisfied, scale up preserving aspect ratio
        - Use mathematical approach to find minimum valid dimensions
        - Ensure final dimensions satisfy both constraints
        
        Args:
            width (int): Original width
            height (int): Original height
            
        Returns:
            tuple: (new_width, new_height) satisfying SALAD constraints
        """
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # First attempt: round to nearest multiples of 14
        initial_width = round(width / 14) * 14
        initial_height = round(height / 14) * 14
        
        # Ensure minimum size (at least 14x14)
        initial_width = max(initial_width, 14)
        initial_height = max(initial_height, 14)
        
        # Check if area constraint is satisfied
        if initial_width * initial_height > self.MIN_AREA - 1:
            print(f"Initial size {initial_width}x{initial_height} satisfies constraints (area: {initial_width * initial_height})")
            return initial_width, initial_height
        
        print(f"Initial size {initial_width}x{initial_height} too small (area: {initial_width * initial_height} <= {self.MIN_AREA - 1})")
        
        # Need to scale up while maintaining aspect ratio and divisibility by 14
        # Calculate minimum dimensions needed
        min_area = self.MIN_AREA
        
        if aspect_ratio >= 1:  # Width >= Height
            # Calculate minimum height, then derive width
            # H * W > min_area, W = H * aspect_ratio
            # H * H * aspect_ratio > min_area
            # H > sqrt(min_area / aspect_ratio)
            min_height = math.sqrt(min_area / aspect_ratio)
            min_width = min_height * aspect_ratio
        else:  # Height > Width
            # Calculate minimum width, then derive height
            # W * H > min_area, H = W / aspect_ratio
            # W * W / aspect_ratio > min_area
            # W > sqrt(min_area * aspect_ratio)
            min_width = math.sqrt(min_area * aspect_ratio)
            min_height = min_width / aspect_ratio
        
        # Round up to next multiple of 14
        target_width = math.ceil(min_width / 14) * 14
        target_height = math.ceil(min_height / 14) * 14
        
        # Double-check constraint (safety measure)
        while target_width * target_height <= self.MIN_AREA - 1:
            # If still not enough, increase the smaller dimension
            if target_width <= target_height:
                target_width += 14
            else:
                target_height += 14
        
        print(f"Scaled to {target_width}x{target_height} to satisfy constraints (area: {target_width * target_height})")
        return target_width, target_height
    
    def _input_transform(self, image_size=None):
        """
        Create input transform for SALAD preprocessing.
        
        Args:
            image_size (tuple): Target image size (width, height)
            
        Returns:
            torchvision.transforms.Compose: Transform pipeline
        """
        if image_size:
            return T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.MEAN, std=self.STD)
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=self.MEAN, std=self.STD)
            ])
    
    def extract_features(self, image_path):
        """
        Extract features from an image using SALAD.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Feature embedding tensor
        """
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original size
        width, height = image.size
        
        # Calculate valid SALAD size
        new_width, new_height = self._get_valid_salad_size(width, height)
        
        # Verify constraints
        area = new_width * new_height
        patches = area / (14 * 14)
        print(f"Original: {width}x{height} -> SALAD: {new_width}x{new_height}")
        print(f"Area: {area}, Patches: {patches:.1f} (constraint: > 64)")
        
        assert patches > 64, f"Constraint violated: {patches} <= 64"
        assert new_width % 14 == 0, f"Width not divisible by 14: {new_width}"
        assert new_height % 14 == 0, f"Height not divisible by 14: {new_height}"
        
        # Create transform with the calculated size
        transform = self._input_transform(image_size=(new_height, new_width))  # Note: PIL uses (width, height), transforms use (height, width)
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            features = self.model(input_tensor)
        
        return features
    
    def extract_features_from_pil(self, pil_image):
        """
        Extract features from a PIL Image object.
        
        Args:
            pil_image (PIL.Image): PIL Image object
            
        Returns:
            torch.Tensor: Feature embedding tensor
        """
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Get original size
        width, height = pil_image.size
        
        # Calculate valid SALAD size
        new_width, new_height = self._get_valid_salad_size(width, height)
        
        # Verify constraints
        area = new_width * new_height
        patches = area / (14 * 14)
        print(f"PIL Image: {width}x{height} -> SALAD: {new_width}x{new_height}")
        print(f"Area: {area}, Patches: {patches:.1f} (constraint: > 64)")
        
        assert patches > 64, f"Constraint violated: {patches} <= 64"
        assert new_width % 14 == 0, f"Width not divisible by 14: {new_width}"
        assert new_height % 14 == 0, f"Height not divisible by 14: {new_height}"
        
        # Create transform with the calculated size
        transform = self._input_transform(image_size=(new_height, new_width))
        
        # Apply transforms
        input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            features = self.model(input_tensor)
        
        return features
