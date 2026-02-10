"""
RGBD Observation Encoder

Extends TimmObsEncoder to handle both RGB and depth images.
Supports multiple fusion strategies:
- 'early': Concatenate depth as 4th channel to RGB (requires modifying first conv layer)
- 'late': Separate encoders for RGB and depth, concatenate features
- 'cross_attention': Use cross-attention between RGB and depth features
"""

import copy
import math
import logging
from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    """Attention pooling for spatial features."""
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class DepthEncoder(nn.Module):
    """
    Simple CNN encoder for depth images.
    Takes single-channel depth and outputs feature map.
    """
    def __init__(self, output_dim: int = 256, downsample_ratio: int = 32):
        super().__init__()
        
        # Simple ResNet-like encoder for depth
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(8, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        if downsample_ratio == 32:
            self.layer4 = self._make_layer(256, output_dim, 2, stride=2)
        else:
            self.layer4 = self._make_layer(256, output_dim, 2, stride=1)
        
        self.output_dim = output_dim
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False))
        layers.append(nn.GroupNorm(8, out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class RGBDObsEncoder(ModuleAttrMixin):
    """
    RGBD Observation Encoder.
    
    Supports three fusion strategies:
    - 'early': Concatenate depth as 4th channel, modify RGB encoder's first conv
    - 'late': Separate encoders, concatenate features at the end
    - 'cross_attention': Use cross-attention between RGB and depth features
    """
    
    def __init__(self,
            shape_meta: dict,
            rgb_model_name: str = 'vit_base_patch16_clip_224.openai',
            rgb_pretrained: bool = True,
            rgb_frozen: bool = False,
            depth_model_name: str = 'resnet18',
            depth_pretrained: bool = False,
            depth_frozen: bool = False,
            fusion_method: str = 'late',  # 'early', 'late', or 'cross_attention'
            global_pool: str = '',
            transforms: list = None,
            use_group_norm: bool = False,
            share_rgb_model: bool = False,
            imagenet_norm: bool = False,
            feature_aggregation: str = 'attention_pool_2d',
            downsample_ratio: int = 32,
            position_encording: str = 'learnable',
            pretrained: bool = None,  # For workspace compatibility, ignored if rgb_pretrained is set
        ):
        """
        Args:
            shape_meta: Dictionary describing observation shapes
            rgb_model_name: Timm model name for RGB encoder
            rgb_pretrained: Use pretrained weights for RGB
            rgb_frozen: Freeze RGB encoder weights
            depth_model_name: Model for depth ('resnet18', 'simple_cnn')
            depth_pretrained: Use pretrained weights for depth
            depth_frozen: Freeze depth encoder weights
            fusion_method: How to combine RGB and depth features
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        rgb_keys = list()
        depth_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        
        # Create RGB model
        rgb_model = timm.create_model(
            model_name=rgb_model_name,
            pretrained=rgb_pretrained,
            global_pool=global_pool,
            num_classes=0
        )
        
        if rgb_frozen:
            for param in rgb_model.parameters():
                param.requires_grad = False
        
        # Determine RGB feature dimension
        rgb_feature_dim = None
        if rgb_model_name.startswith('resnet'):
            if downsample_ratio == 32:
                modules = list(rgb_model.children())[:-2]
                rgb_model = torch.nn.Sequential(*modules)
                rgb_feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(rgb_model.children())[:-3]
                rgb_model = torch.nn.Sequential(*modules)
                rgb_feature_dim = 256
        elif rgb_model_name.startswith('vit'):
            rgb_feature_dim = 768  # ViT base
        elif rgb_model_name.startswith('convnext'):
            if downsample_ratio == 32:
                modules = list(rgb_model.children())[:-2]
                rgb_model = torch.nn.Sequential(*modules)
                rgb_feature_dim = 1024
        
        if use_group_norm and not rgb_pretrained:
            rgb_model = replace_submodules(
                root_module=rgb_model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )
        
        # Create depth model
        depth_feature_dim = 256
        if depth_model_name == 'simple_cnn':
            depth_model = DepthEncoder(output_dim=depth_feature_dim, downsample_ratio=downsample_ratio)
        elif depth_model_name.startswith('resnet'):
            depth_model = timm.create_model(
                model_name=depth_model_name,
                pretrained=depth_pretrained,
                in_chans=1,  # Single channel for depth
                global_pool=global_pool,
                num_classes=0
            )
            if downsample_ratio == 32:
                modules = list(depth_model.children())[:-2]
                depth_model = torch.nn.Sequential(*modules)
                if '18' in depth_model_name or '34' in depth_model_name:
                    depth_feature_dim = 512
                else:
                    depth_feature_dim = 2048
        else:
            depth_model = DepthEncoder(output_dim=depth_feature_dim, downsample_ratio=downsample_ratio)
        
        if depth_frozen:
            for param in depth_model.parameters():
                param.requires_grad = False
        
        # Determine image shape from shape_meta
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        
        # Store config for synchronized transforms
        self.imagenet_norm = imagenet_norm
        self.crop_ratio = None
        self.image_size = image_shape[0] if image_shape else 224
        
        # Create transforms list (geometric + color augmentation for RGB)
        # Note: We'll apply geometric transforms manually in forward() for synchronization
        rgb_color_transforms = []
        if transforms is not None:
            if not isinstance(transforms[0], torch.nn.Module):
                assert transforms[0].type == 'RandomCrop'
                self.crop_ratio = transforms[0].ratio
                # Skip the RandomCrop config, keep rest (ColorJitter, etc.)
                rgb_color_transforms = [t for t in transforms[1:] if isinstance(t, torch.nn.Module)]
            else:
                # Already module transforms - filter to color-only
                for t in transforms:
                    if isinstance(t, (torchvision.transforms.ColorJitter,)):
                        rgb_color_transforms.append(t)
        
        # Add ImageNet normalization if enabled (required for pretrained ViT!)
        if imagenet_norm:
            rgb_color_transforms.append(
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        # RGB transform: color jitter + imagenet norm (geometric done in forward())
        transform = nn.Identity() if len(rgb_color_transforms) == 0 else torch.nn.Sequential(*rgb_color_transforms)
        
        # Depth transform: no color jitter, no imagenet norm (geometric done in forward())
        depth_transform = nn.Identity()
        
        # Process shape_meta
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            
            if type == 'rgb':
                rgb_keys.append(key)
                this_model = rgb_model if share_rgb_model else copy.deepcopy(rgb_model)
                key_model_map[key] = this_model
                key_transform_map[key] = transform
            elif type == 'depth':
                depth_keys.append(key)
                key_model_map[key] = copy.deepcopy(depth_model)
                key_transform_map[key] = depth_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
        
        rgb_keys = sorted(rgb_keys)
        depth_keys = sorted(depth_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:       ', rgb_keys)
        print('depth keys:     ', depth_keys)
        print('low_dim_keys:   ', low_dim_keys)
        
        self.rgb_model_name = rgb_model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation
        self.rgb_feature_dim = rgb_feature_dim
        self.depth_feature_dim = depth_feature_dim
        
        # Feature aggregation for spatial features
        if rgb_model_name.startswith('vit'):
            self.feature_aggregation = None
        
        # Initialize attention pools (always create them, but may not use them)
        self.rgb_attention_pool = None
        self.depth_attention_pool = None
        
        if self.feature_aggregation == 'attention_pool_2d':
            self.rgb_attention_pool = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=rgb_feature_dim,
                num_heads=rgb_feature_dim // 64,
                output_dim=rgb_feature_dim
            )
            self.depth_attention_pool = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=depth_feature_dim,
                num_heads=max(1, depth_feature_dim // 64),
                output_dim=depth_feature_dim
            )
        
        # Fusion layer for 'late' fusion
        if fusion_method == 'late':
            total_feature_dim = rgb_feature_dim + depth_feature_dim
            # Keep full dimension to preserve depth information
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_feature_dim, total_feature_dim),
                nn.ReLU(),
                nn.Linear(total_feature_dim, total_feature_dim)
            )
            self.output_feature_dim = total_feature_dim  # 1280 instead of 768
        elif fusion_method == 'cross_attention':
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=rgb_feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.depth_proj = nn.Linear(depth_feature_dim, rgb_feature_dim)
            self.output_feature_dim = rgb_feature_dim
        else:
            self.output_feature_dim = rgb_feature_dim
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature, attention_pool, is_vit=False):
        """Aggregate spatial features into a single vector."""
        if is_vit:
            # ViT outputs (B, seq_len, features), use CLS token
            assert len(feature.shape) == 3, f"Expected 3D tensor for ViT, got shape {feature.shape}"
            return feature[:, 0, :]
        
        assert len(feature.shape) == 4, f"Expected 4D tensor for CNN, got shape {feature.shape}"
        if self.feature_aggregation == 'attention_pool_2d' and attention_pool is not None:
            return attention_pool(feature)
        
        # Default: global average pooling
        return feature.mean(dim=[2, 3])
    
    def _apply_synchronized_geometric_transforms(self, rgb_img, depth_img):
        """
        Apply the same random crop and resize to both RGB and depth.
        This ensures spatial alignment between modalities.
        """
        if self.crop_ratio is None or not self.training:
            # No cropping or in eval mode - just return as-is
            return rgb_img, depth_img
        
        # Get crop parameters (same for both)
        crop_size = int(self.image_size * self.crop_ratio)
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            rgb_img, output_size=(crop_size, crop_size)
        )
        
        # Apply same crop to both
        rgb_img = torchvision.transforms.functional.crop(rgb_img, i, j, h, w)
        rgb_img = torchvision.transforms.functional.resize(rgb_img, [self.image_size, self.image_size], antialias=True)
        
        if depth_img is not None:
            depth_img = torchvision.transforms.functional.crop(depth_img, i, j, h, w)
            depth_img = torchvision.transforms.functional.resize(depth_img, [self.image_size, self.image_size], antialias=True)
        
        return rgb_img, depth_img
    
    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        rgb_features = []
        depth_features = []
        B = batch_size
        T = 1  # Default, will be updated
        
        # Collect all RGB and depth images first
        rgb_images = {}
        depth_images = {}
        
        for key in self.rgb_keys:
            img = obs_dict[key]
            if len(img.shape) == 5:
                B, T = img.shape[:2]
                img = img.reshape(B*T, *img.shape[2:])
            elif len(img.shape) == 4:
                BT = img.shape[0]
                B = batch_size
                T = BT // B
            rgb_images[key] = img
        
        for key in self.depth_keys:
            if key not in obs_dict:
                continue
            depth = obs_dict[key]
            if len(depth.shape) == 5:
                B, T = depth.shape[:2]
                depth = depth.reshape(B*T, *depth.shape[2:])
            elif len(depth.shape) == 4:
                BT = depth.shape[0]
                B = batch_size
                T = BT // B
            depth_images[key] = depth
        
        # Apply synchronized geometric transforms to paired RGB-depth
        # Assume camera0_rgb pairs with camera0_depth, etc.
        for rgb_key in self.rgb_keys:
            depth_key = rgb_key.replace('_rgb', '_depth')
            rgb_img = rgb_images[rgb_key]
            depth_img = depth_images.get(depth_key)
            
            # Apply same random crop to both
            rgb_img, depth_img = self._apply_synchronized_geometric_transforms(rgb_img, depth_img)
            rgb_images[rgb_key] = rgb_img
            if depth_img is not None and depth_key in depth_images:
                depth_images[depth_key] = depth_img
        
        # Process RGB inputs (color transforms + encoding)
        for key in self.rgb_keys:
            img = rgb_images[key]
            
            # Apply color transforms (ColorJitter + ImageNet normalization)
            img = self.key_transform_map[key](img)
            
            raw_feature = self.key_model_map[key](img)
            is_vit = self.rgb_model_name.startswith('vit')
            feature = self.aggregate_feature(raw_feature, self.rgb_attention_pool, is_vit=is_vit)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T, f"RGB feature shape: {feature.shape}, expected ({B*T}, ?)"
            rgb_features.append(feature.reshape(B, T, -1))
        
        # Process depth inputs
        for key in self.depth_keys:
            if key not in depth_images:
                continue
            depth = depth_images[key]
            
            # No color transforms for depth, just encode
            raw_feature = self.key_model_map[key](depth)
            feature = self.aggregate_feature(raw_feature, self.depth_attention_pool, is_vit=False)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T, f"Depth feature shape: {feature.shape}, expected ({B*T}, ?)"
            depth_features.append(feature.reshape(B, T, -1))
        
        # Fuse RGB and depth features
        if len(rgb_features) > 0 and len(depth_features) > 0:
            # Average across all RGB/depth cameras if multiple
            rgb_feat = torch.stack(rgb_features, dim=0).mean(dim=0)  # B, T, D_rgb
            depth_feat = torch.stack(depth_features, dim=0).mean(dim=0)  # B, T, D_depth
            
            if self.fusion_method == 'late':
                # Concatenate and project
                combined = torch.cat([rgb_feat, depth_feat], dim=-1)
                fused = self.fusion_layer(combined)
                features.append(fused.reshape(B, -1))
            elif self.fusion_method == 'cross_attention':
                # Use depth to attend to RGB
                depth_proj = self.depth_proj(depth_feat)
                attended, _ = self.cross_attention(depth_proj, rgb_feat, rgb_feat)
                fused = rgb_feat + attended
                features.append(fused.reshape(B, -1))
            else:
                # Just concatenate
                features.append(rgb_feat.reshape(B, -1))
                features.append(depth_feat.reshape(B, -1))
        else:
            # Only RGB or only depth
            for feat in rgb_features:
                features.append(feat.reshape(batch_size, -1))
            for feat in depth_features:
                features.append(feat.reshape(batch_size, -1))
        
        # Process low-dim inputs
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
        
        # Concatenate all features
        result = torch.cat(features, dim=-1)
        
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape
