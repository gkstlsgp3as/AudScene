import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F

class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_audio_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))

    def forward(self, audio_embeddings, mask):
        B, N, _ = audio_embeddings.shape # B: batch_size, N: num_tokens, C: self.in_dim (=768)
        audio_null = self.null_audio_feature.view(1, 1, -1) # 1*1*C
        audio_embeddings = mask * audio_embeddings + (1 - mask) * audio_null

        objs = self.linears(audio_embeddings)
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs