

import torch
from torch import nn
from mmengine import Config

# We now import from the registry rather than from mmaction.models
from mmaction.registry import MODELS




class TemporalSegmentAttention(nn.Module):
    """Learn a normalized importance weight for each sampled TSN segment.

    Input: [B, T, D] per-segment features; output: pooled [B, D] and [B, T] weights.
    The feature dimension D is inferred from the configured TSN classifier.
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError('feature_dim and hidden_dim must be positive')
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, segment_features):
        if segment_features.ndim != 3:
            raise ValueError(f'Expected [B,T,D], got {tuple(segment_features.shape)}')
        if segment_features.shape[1] < 1:
            raise ValueError('At least one temporal segment is required')
        weights = torch.softmax(self.score(segment_features).squeeze(-1), dim=1)
        pooled = torch.sum(segment_features * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class TSNWithYOLO(nn.Module):
    def __init__(
            self,
            mmaction_cfg_path,
            yolo_encoding_dim=128,
            device='cuda',
            freeze_tsn_backbone=False,
            temporal_attention=False,
            temporal_attention_hidden_dim=128
    ):
        """
        :param mmaction_cfg_path: path to a single-file config for TSN in MMAction2 style
        :param yolo_encoding_dim: dimension of YOLO's encoded features to fuse
        :param device: 'cuda' or 'cpu'
        :param freeze_tsn_backbone: if True, sets requires_grad=False on the TSN backbone
        """
        super().__init__()
        self.device = device

        # 1) Parse the config
        config = Config.fromfile(mmaction_cfg_path)
        # Insert train_cfg/test_cfg into model config if needed
        # (some config variants already contain them, so we do it safely)
        if 'train_cfg' not in config.model:
            config.model['train_cfg'] = config.get('train_cfg', None)
        if 'test_cfg' not in config.model:
            config.model['test_cfg'] = config.get('test_cfg', None)

        # 2) Build TSN from the registry
        self.tsn = MODELS.build(config.model)
        self.tsn.to(self.device)

        # 3) Optionally freeze TSN backbone
        if freeze_tsn_backbone:
            for param in self.tsn.backbone.parameters():
                param.requires_grad = False

        # 4) Create a fusion layer that appends YOLO encoding to TSN's final features
        #    We assume TSN has self.tsn.cls_head.fc_cls = final FC
        original_fc = self.tsn.cls_head.fc_cls
        in_features = original_fc.in_features   # e.g. 2048 for ResNet50
        out_features = original_fc.out_features

        self.fusion_fc = nn.Linear(in_features + yolo_encoding_dim, out_features)

        # Copy original fc_cls weights & bias into the corresponding portion of fusion_fc
        with torch.no_grad():
            self.fusion_fc.weight[:, :in_features] = original_fc.weight
            self.fusion_fc.bias[:] = original_fc.bias

        self.temporal_attention_enabled = bool(temporal_attention)
        self.temporal_attn = (
            TemporalSegmentAttention(in_features, hidden_dim=temporal_attention_hidden_dim)
            if self.temporal_attention_enabled else None
        )

    def forward(self, frames, yolo_encodings, return_attention=False):
        """
        frames:        shape [B, T, 3, H, W]
        yolo_encodings shape [B, T, yolo_encoding_dim], e.g. 128
        Returns:       shape [B, num_classes]
        """
        B, T, C, H, W = frames.shape

        # 1) Run TSN backbone
        #    Flatten temporal dimension => [B*T, 3, H, W]
        frames_reshape = frames.view(-1, C, H, W)   # => [B*T, 3, H, W]
        feat = self.tsn.backbone(frames_reshape)    # => [B*T, feat_dim, ...]
        # Global average pool (assuming TSN's cls_head has avg_pool)
        feat = self.tsn.cls_head.avg_pool(feat)     # => [B*T, feat_dim, 1, 1]
        feat = feat.view(feat.size(0), -1)          # => [B*T, feat_dim]

        # 2) Temporal fusion for TSN
        feat = feat.view(B, T, -1)
        if yolo_encodings.ndim != 3 or yolo_encodings.shape[:2] != (B, T):
            raise ValueError(f'YOLO encodings must be [B,T,D] aligned with frames [B,T,...]; got {tuple(yolo_encodings.shape)}')

        # Baseline uses uniform mean pooling. Enhanced mode learns segment weights
        # from TSN features and applies the same weights to aligned YOLO features.
        if self.temporal_attn is None:
            temporal_weights = feat.new_full((B, T), 1.0 / T)
            tsn_feat = feat.mean(dim=1)
        else:
            tsn_feat, temporal_weights = self.temporal_attn(feat)
        yolo_feat = torch.sum(yolo_encodings * temporal_weights.unsqueeze(-1), dim=1)

        # 4) Concat TSN + YOLO => [B, feat_dim + yolo_dim]
        fused_feat = torch.cat([tsn_feat, yolo_feat], dim=1)

        # 5) Classify
        logits = self.fusion_fc(fused_feat)  # => [B, num_classes]
        return (logits, temporal_weights) if return_attention else logits
