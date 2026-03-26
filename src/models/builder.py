import torch.nn as nn

from src.models.CNN import AudioSpectrogramCNN
from src.models.CNN_DenseNet_121 import CNN_DenseNet_121
from src.models.CNN_MultiFeatureFusionAttention import BaselineMultiFeatureCNN, MultiFeatureFusionAttentionCNNLogits, ModelConfig

# Model imports mapping
def build_model(backbone: str, in_ch: int, num_classes: int, model_cfg: dict) -> nn.Module:
    name = str(backbone).strip().lower()
    dropout = float(model_cfg.get("dropout", 0.3))
    fc_dim = int(model_cfg.get("fc_hidden_dim", 256))
    attn_red = int(model_cfg.get("attention_reduction", 8))


    if name == "cnn": return AudioSpectrogramCNN(in_ch=in_ch, num_classes=num_classes, p_drop=dropout)
    if name == "cnn_densenet_121": return CNN_DenseNet_121(in_ch=in_ch, num_classes=num_classes, p_drop=dropout)
    
    cfg = ModelConfig(in_channels=in_ch, num_classes=num_classes, fc_hidden_dim=fc_dim, attention_reduction=attn_red, dropout=dropout)
    if "baseline" in name: return BaselineMultiFeatureCNN(cfg)
    if "fusion" in name: return MultiFeatureFusionAttentionCNNLogits(cfg)
    raise ValueError(f"Unsupported backbone: {backbone}")
