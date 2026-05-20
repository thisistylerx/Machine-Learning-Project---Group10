import os
import json
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 80
LR = 5e-4
PATIENCE = 10

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RESULT_ROOT = os.path.join(PROJECT_ROOT, 'results_v2')
os.makedirs(RESULT_ROOT, exist_ok=True)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CoSupFormer(nn.Module):
    def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyConv1d(embed_dim, 3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(64)
        )

        self.pos_enc = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256,
            dropout=0.3, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)