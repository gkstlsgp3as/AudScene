import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import librosa
import laion_clap
import numpy as np
from transformers import CLIPTokenizer, CLIPModel
from tqdm import tqdm
import logging

# ---- Define logger ----------------------- #
os.makedirs("/data2/jungwon/AudScene/contrastive_learned_CLAP7/", exist_ok=True)
# logging.basicConfig(filename='/data2/jungwon/AudScene/contrastive_learned_CLAP3/train.log')
def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
logger = get_logger("/data2/jungwon/AudScene/contrastive_learned_CLAP7/train.log")

device = "cuda"
logger.info("Contrastive Learning started!!")
# --------------------------- #

# ---- Construct name_category ----------------------- #
# Load name_category cse file
csv_path = "/data2/VGGSound/vggsound_path.csv"
with open(csv_path, 'r') as f:
    name_category = f.readlines()

#  (Added) name_category -> small_name_category ------ #
csv_path2 = "/data2/VGGSound/vggsound_mag20_aud40_caption_HPS_bbox_1218.csv"
with open(csv_path2, 'r') as f:
    small = f.readlines()

small_list = []
for i in range(len(small)):
    small_list.append(small[i].split(',')[0] + '.jpg')

idx_list = []
for i in range(len(name_category)):
    if os.path.basename(name_category[i].split(',')[1]) in small_list:
        idx_list.append(i)

small_name_category = []
for idx in idx_list:
    small_name_category.append(name_category[idx])

name_category = small_name_category

# Make a list of category
category_list = []
for i in range(len(name_category)):
    category_list.append(name_category[i].split(',')[2:])
for idx, category in enumerate(category_list):
    category_list[idx] = ','.join(category)
for idx, _ in enumerate(category_list):
    while True:
        if '\"' not in category_list[idx]:
            break
        else:
            category_list[idx] = category_list[idx].replace('\"', "")
    while True:
        if '\n' not in category_list[idx]:
            break
        else: 
            category_list[idx] = category_list[idx].replace('\n', "")
category_list_non_repeat = list(set(category_list))
# --------------------------- #

def recursively_read_audios(rootdir, must_contain, exts=["wav"]): # please check the file extension
    """
    Your dataset must be organized as [rootdir - subdirectories - audio files].
        If your dataset is organized as [rootdir - audio files], please use 
        "dataset/dataset_sem.py/recursively_read" instead.
    
    out: A list of audio_file_paths
    """
    out = [] 
    for r, sub_dirs, _ in os.walk(rootdir):
        for sub_dir in sub_dirs:
            for d, _, f in os.walk(os.path.join(r, sub_dir)):
                for file in f:
                    if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(d, file)):
                        out.append(os.path.join(d, file))
    return out


# ---- Construct audio files ----------------------- #
audio_rootdir = "/data2/VGGSound/audio_mag20_aud40_caption_HPS_bbox_1218/train/"

audio_files = recursively_read_audios(rootdir=audio_rootdir, must_contain="")
audio_files.sort()
# --------------------------- #


# ---- Load name_dict ----------------------- #
name_dict_path = "/data2/jungwon/AudScene/name_dict_1218.json"

with open(name_dict_path, 'r') as f: 
    name_dict = json.load(f)
# --------------------------- #

# ---- Load pre-trained audio and text encoders ----------------------- #
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()
        self.transformer = self.transformer.to(device)

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, return_pooler_output=False):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer.get_text_features(input_ids=tokens).to(device=self.device)
        
        return outputs

    def encode(self, text, return_pooler_output=False):
        return self(text, return_pooler_output)

audio_encoder = laion_clap.CLAP_Module(enable_fusion=False)
audio_encoder.load_ckpt()
text_encoder = FrozenCLIPEmbedder()

# --------------------------- #

# ---- Define dataset and train data loader ----------------------- #
class ContrastiveDataset():
    def __init__(self, audio_rootdir, name_dict_path, prompt_emb_dict):
        self.name_dict_path = name_dict_path
        self.audio_files = audio_rootdir

        audio_files = recursively_read_audios(rootdir=audio_rootdir, must_contain="")
        audio_files.sort()

        self.audio_files = audio_files

        with open(name_dict_path, 'r') as f:
            name_dict = json.load(f)
        
        self.name_dict = name_dict
        self.prompt_emb_dict = prompt_emb_dict
    
    def __getitem__(self, index):
        out = {}

        out['id'] = index
        out['audio_path'] = str(self.audio_files[index])

        positive_prompt = "A photo of {}.".format(self.name_dict[os.path.basename(self.audio_files[index])])
        positive_embedding = self.prompt_emb_dict[positive_prompt].squeeze(dim=0)

        dict_keys = list(prompt_emb_dict.keys())
        negative_embeddings = torch.cat([prompt_emb_dict[prompt] for prompt in dict_keys if prompt != positive_prompt], dim=0)

        out['positive_embedding'] = positive_embedding # [768]
        out['negative_embeddings'] = negative_embeddings # [22, 768]

        return out
    
    def __len__(self):
        return len(self.audio_files)

# A dictionary of 21 prompt embeddings
unique_items = list(set(name_dict.values()))
prompt_emb_dict = dict()
for item in unique_items:
    prompt_emb_dict["A photo of {}.".format(item)] = None

for key in prompt_emb_dict.keys():
    prompt_emb_dict[key] = text_encoder(key)

contrast_dataset = ContrastiveDataset(audio_rootdir, name_dict_path, prompt_emb_dict)
loader_train = DataLoader(contrast_dataset, batch_size=64, shuffle=True)
# --------------------------- #

def get_audio_embeddings_with_projection(audio_encoder, batch, projection_layer):
    """
    This function is used to extract audio embeddings from audio files.
    Output shape: [batch_size, num_tokens, hidden_dim]
    """
    B = len(batch['audio_path'])
    batch_audio_path = batch['audio_path'] # A list of shape [B]

    ### Your code here ---------------------------------------- ###
    audio_embeddings = audio_encoder.get_audio_embedding_from_filelist(x = batch_audio_path, use_tensor=True)
    audio_embeddings = projection_layer(audio_embeddings)
    ### -------------------------------------------------------- ###

    return audio_embeddings

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    [From https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py]
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

# --------------------------- #
def validate_in_test(val_audio_rootdir, name_category, prompt_emb_dict, category_list):
    val_audio_files = recursively_read_audios(rootdir=val_audio_rootdir, must_contain="")
    val_audio_files.sort()
    
    avg = 0
    total_iter = len(val_audio_files) // 10
    for iter in range(total_iter):
        test_batch = dict()
        audio_paths = []
        category_names =[]
        for j in range(iter*10, (iter+1)*10):
            for idx in range(len(name_category)):
                if os.path.basename(val_audio_files[j]).split('.')[0] == os.path.basename(name_category[idx].split(',')[1]).split('.')[0]:
                    category_name = category_list[idx]
                    break
            audio_paths.append(val_audio_files[j])
            category_names.append(category_name)

        test_batch['audio_path'] = audio_paths
        audio_embeddings = get_audio_embeddings_with_projection(audio_encoder, test_batch, projection_layer)

        prompt_embeddings = []
        for name in category_names:
            prompt = "A photo of {}.".format(name)
            prompt_embeddings.append(prompt_emb_dict[prompt])
        prompt_embeddings = torch.concat(prompt_embeddings, dim=0)

        cosine_similarities = F.cosine_similarity(audio_embeddings, prompt_embeddings, dim=1).mean()
        avg += cosine_similarities.detach().to("cpu").item()
    avg /= total_iter
    return avg


# ---- Training ----------------------- #
projection_layer = nn.Linear(in_features=512, out_features=768).to(device)
init.xavier_uniform_(projection_layer.weight)
init.zeros_(projection_layer.bias)

def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


optimizer = torch.optim.Adam(list(audio_encoder.parameters()) + list(projection_layer.parameters()), lr=1e-4)
loss = InfoNCE(negative_mode='paired')
num_epochs = 201


for epoch in range(num_epochs):
    for idx, batch in enumerate(tqdm(loader_train)):
        # Move the batch to teh device
        batch_to_device(batch, device)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Get the audio embeddings
        audio_embeddings = get_audio_embeddings_with_projection(audio_encoder, batch, projection_layer)
        
        # Get the positive and negative keys for constrastive learning
        positive_key = batch['positive_embedding']
        negative_keys = batch['negative_embeddings']

        # Compute the loss
        loss_value = loss(audio_embeddings, positive_key, negative_keys)

        # Backpropagate the loss
        loss_value.backward()

        # Update the weights
        optimizer.step()
    print("Epoch: {}, Loss: {}".format(epoch, loss_value.item()))
    if epoch % 5 == 0:
        val_audio_rootdir = "/data2/VGGSound/audio_mag20_aud40_caption_HPS_bbox_1218/test/"
        valid_score = validate_in_test(val_audio_rootdir, name_category, prompt_emb_dict, category_list)
        logging.info("Epoch: {}, Validation cosine similarity: {}".format(epoch, valid_score))
        torch.save({
            'model_state_dict': audio_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'projection_layer_state_dict': projection_layer.state_dict(),
            }, '/data2/jungwon/AudScene/contrastive_learned_CLAP7/checkpoint_epoch_{}.pth'.format(epoch))
# --------------------------- #
"""
[Before you run]
Change saved_path: Line 429

[If you change some files, please check the path for]
csv_path, csv_path2, audio_rootdir, name_dict_path, val_audio_rootdir
"""