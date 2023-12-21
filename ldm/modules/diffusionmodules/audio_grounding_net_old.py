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

        # self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        # self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_audio_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        # self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, audio_embeddings, mask):
        B, N, _ = audio_embeddings.shape # B: batch_size, N: num_tokens, C: self.in_dim
        audio_null = self.null_audio_feature.view(1, 1, -1) # 1*1*C
        mask = mask.view(mask.shape[0], 1, 1) # B*1*1
        audio_embeddings = mask * audio_embeddings + (1 - mask) * audio_null

        # xyxy_embedding = self.fourier_embedder(bbox) # B*N*4 --> B*N*(4*fourier_freqs*2)
        # xyxy_null = self.null_position_feature.view(1, 1, -1) # 1*1*(4*fourier_freqs*2)
        # xyxy_embedding = mask * xyxy_embedding + (1 - mask) * xyxy_null

        # objs = self.linears(torch.cat([audio_embeddings, xyxy_embedding], dim=-1))

        objs = self.linears(audio_embeddings)
        # assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs