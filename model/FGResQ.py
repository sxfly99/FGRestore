import numpy as np
import timm
import torch
import torchvision
import torch.nn as nn
from transformers import CLIPVisionModel
# import open_clip
import torchvision.transforms as transforms
from PIL import Image

def load_clip_model(clip_model="openai/ViT-B-16", clip_freeze=True, precision='fp16'):
    pretrained, model_tag = clip_model.split('/')
    pretrained = None if pretrained == 'None' else pretrained
    # clip_model = open_clip.create_model(model_tag, precision=precision, pretrained=pretrained)
    # clip_model = timm.create_model('timm/vit_base_patch16_clip_224.openai', pretrained=True, in_chans=3)
    clip_model = CLIPVisionModel.from_pretrained(clip_model)
    if clip_freeze:
        for param in clip_model.parameters():
            param.requires_grad = False

    if model_tag == 'clip-vit-base-patch16':
        feature_size = dict(global_feature=768, local_feature=[196, 768])
    elif model_tag == 'ViT-L-14-quickgelu' or model_tag == 'ViT-L-14':
        feature_size = dict(global_feature=768, local_feature=[256, 1024])
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")

    return clip_model, feature_size

class DualBranch(nn.Module):

    def __init__(self, clip_model="openai/ViT-B-16", clip_freeze=True, precision='fp16'):
        super(DualBranch, self).__init__()
        self.clip_freeze = clip_freeze

        # Load CLIP model
        self.clip_model, feature_size = load_clip_model(clip_model, clip_freeze, precision)

        # Initialize CLIP vision model for task classification
        self.task_cls_clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        

        self.head = nn.Linear(feature_size['global_feature']*3, 1)
        self.compare_head =nn.Linear(feature_size['global_feature']*6, 3)
        
    
        self.prompt = nn.Parameter(torch.rand(1, feature_size['global_feature']))
        self.task_mlp = nn.Sequential(
            nn.Linear(feature_size['global_feature'], feature_size['global_feature']), 
            nn.SiLU(False),
            nn.Linear(feature_size['global_feature'], feature_size['global_feature']))
        self.prompt_mlp = nn.Linear(feature_size['global_feature'], feature_size['global_feature'])
        
        with torch.no_grad():
            self.task_mlp[0].weight.fill_(0.0)
            self.task_mlp[0].bias.fill_(0.0)
            self.task_mlp[2].weight.fill_(0.0)
            self.task_mlp[2].bias.fill_(0.0)
            self.prompt_mlp.weight.fill_(0.0)
            self.prompt_mlp.bias.fill_(0.0)
        
        # Load pre-trained weights
        self._load_pretrained_weights("weights/Degradation.pth")


        for param in self.task_cls_clip.parameters():
            param.requires_grad = False

        # Unfreeze the last two layers
        # for i in range(10, 12):  # Layers 10 and 11
        #     for param in self.task_cls_clip.vision_model.encoder.layers[i].parameters():
        #         param.requires_grad = True
    def _load_pretrained_weights(self, state_dict_path):
        """
        Load pre-trained weights, including the CLIP model and classification head.
        """
        # Load state dictionary
        state_dict = torch.load(state_dict_path)
        
        # Separate weights for CLIP model and classification head
        clip_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith('clip_model.'):
                # Remove 'clip_model.' prefix for the CLIP model
                new_key = key.replace('clip_model.', '')
                clip_state_dict[new_key] = value
            # elif key in ['head.weight', 'head.bias']:
            #     # Save weights for the classification head
            #     head_state_dict[key] = value
        
        # Load weights for the CLIP model
        self.task_cls_clip.load_state_dict(clip_state_dict, strict=False)
        print("Successfully loaded CLIP model weights")
        
    def forward(self, x0, x1 = None):
        # features, _ = self.clip_model.encode_image(x)
        if x1 is None:
            # Image features
            features0 = self.clip_model(x0)['pooler_output']
            # Classification features
            task_features0 = self.task_cls_clip(x0)['pooler_output']

            # Learn classification features
            task_embedding = torch.softmax(self.task_mlp(task_features0), dim=1) * self.prompt
            task_embedding = self.prompt_mlp(task_embedding)

            # features = torch.cat([features0, task_features], dim
            features0 = torch.cat([features0, task_embedding, features0+task_embedding], dim=1)
            quality = self.head(features0)
            quality = nn.Sigmoid()(quality)

            return quality, None, None
        elif x1 is not None:
            # features_, _ = self.clip_model.encode_image(x_local)
            # Image features
            features0 = self.clip_model(x0)['pooler_output']
            features1 = self.clip_model(x1)['pooler_output']
            # Classification features
            task_features0 = self.task_cls_clip(x0)['pooler_output']
            task_features1 = self.task_cls_clip(x1)['pooler_output']

            task_embedding0 = torch.softmax(self.task_mlp(task_features0), dim=1) * self.prompt
            task_embedding0 = self.prompt_mlp(task_embedding0)
            task_embedding1 = torch.softmax(self.task_mlp(task_features1), dim=1) * self.prompt
            task_embedding1 = self.prompt_mlp(task_embedding1)

            features0 = torch.cat([features0, task_embedding0, features0+task_embedding0], dim=1)
            features1 = torch.cat([features1, task_embedding1, features1+task_embedding1], dim=1)

            # features0 = torch.cat([features0, task_features0], dim=
            # import pdb; pdb.set_trace()
            features = torch.cat([features0, features1], dim=1)
            # features = torch.cat([features0, features1], dim=1)
            compare_quality = self.compare_head(features)

            # quality0 = self.head(features0)
            # quality1 = self.head(features1)
            quality0 = self.head(features0)
            quality1 = self.head(features1)
            quality0 = nn.Sigmoid()(quality0)
            quality1 = nn.Sigmoid()(quality1)

            # quality = {'quality0': quality0, 'quality1': quality1}

            return quality0, quality1, compare_quality

class FGResQ:
    def __init__(self, model_path, clip_model="openai/clip-vit-base-patch16", input_size=224, device=None):
        """
        Initializes the inference model.

        Args:
            model_path (str): Path to the pre-trained model checkpoint (.pth or .safetensors).
            clip_model (str): Name of the CLIP model to use.
            input_size (int): Input image size for the model.
            device (str, optional): Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")

        # Load the model
        self.model = DualBranch(clip_model=clip_model, clip_freeze=True, precision='fp32')
        
        # Load model weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle different ways of saving state_dict
            if 'model' in state_dict:
                state_dict = state_dict['model']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove 'module.' prefix if it exists
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)
            print(f"Model weights loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def _preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    @torch.no_grad()
    def predict_single(self, image_path):
        """
        Predict the quality score of a single image.
        """
        image_tensor = self._preprocess_image(image_path)
        if image_tensor is None:
            return None

        quality_score, _, _ = self.model(image_tensor)
        return quality_score.squeeze().item()

    @torch.no_grad()
    def predict_pair(self, image_path1, image_path2):
        """
        Compare the quality of two images.
        """
        image_tensor1 = self._preprocess_image(image_path1)
        image_tensor2 = self._preprocess_image(image_path2)

        if image_tensor1 is None or image_tensor2 is None:
            return None

        quality1, quality2, compare_result = self.model(image_tensor1, image_tensor2)
        
        quality1 = quality1.squeeze().item()
        quality2 = quality2.squeeze().item()
        
        # Interpret the comparison result
        compare_probs = torch.softmax(compare_result, dim=-1).squeeze().cpu().numpy()
        prediction = np.argmax(compare_probs)
        
        comparison_map = {0: 'Image 1 is better', 1: 'Image 2 is better', 2: 'Images are of similar quality'}
        
        return {
            'comparison': comparison_map[prediction],
            'comparison_raw': compare_probs.tolist()}
        