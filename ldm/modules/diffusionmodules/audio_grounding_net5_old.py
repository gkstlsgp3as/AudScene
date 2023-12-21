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

        self.linears_1 = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linears_2 = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linears_3 = nn.Sequential(
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

        # objs_1 = self.linears_1(torch.cat([audio_embeddings, xyxy_embedding], dim=-1))
        # objs_2 = self.linears_2(torch.cat([audio_embeddings, xyxy_embedding], dim=-1))
        # objs_3 = self.linears_3(torch.cat([audio_embeddings, xyxy_embedding], dim=-1))
        objs_1 = self.linears_1(audio_embeddings)
        objs_2 = self.linears_2(audio_embeddings)
        objs_3 = self.linears_3(audio_embeddings)
        B, N, C = objs_1.shape
        objs = torch.stack([objs_1, objs_2, objs_3], dim=2).view(B, N * 3, C)

        # assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs