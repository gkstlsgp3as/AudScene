import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from dataset.catalog import DatasetCatalog
from dataset.concat_dataset import ConCatDataset
from clap_modules import Adapt_CLAP_Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import librosa
import laion_clap

device = "cuda"

def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=1000, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas



def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config

def recursively_read_images(rootdir, must_contain="", exts=["png", "jpg", "JPEG", "jpeg"]):
    """
    out: A list of image_file_paths
    """
    out = [] 
    for r, sub_dirs, _ in os.walk(rootdir):
        for sub_dir in sub_dirs:
            for d, _, f in os.walk(os.path.join(r, sub_dir)):
                for file in f:
                    if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(d, file)):
                        out.append(os.path.join(d, file))
    return out

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

def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


@torch.no_grad()
def get_audio_embeddings(batch, audio_encoder):
    """
    This function is used to extract audio embeddings from audio files.
    Output shape: [batch_size, num_tokens, hidden_dim]
    """
    B = len(batch['audio_path'])
    batch_audio_path = batch['audio_path'] # A list of shape [B]

    ### Your code here ---------------------------------------- ###
    audio_embeddings = audio_encoder.get_audio_embedding_from_filelist(x = batch_audio_path, use_tensor=True)
    audio_embeddings = audio_embeddings.unsqueeze(1)  # [B, N (=1), C]
    ### -------------------------------------------------------- ###

    return audio_embeddings


@torch.no_grad()
def run(config, starting_noise=None):
    # - - - - - Make a folder and subfolders to save images - - - - - #
    # Load name_category cse file
    csv_path = config.csv_path      # "/data2/VGGSound/vggsound_path.csv"
    with open(csv_path, 'r') as f:
        name_category = f.readlines()

    #  (Added) name_category -> small_name_category ------ #
    csv_path2 = config.csv_path2     # "/data2/jungwon/vggsound_small_with_class.csv"
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
    # --------------------------- #

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

    # Make a directory for each category
    dir_path = os.path.join("/data2/jungwon/AudScene/generated_images", config.name) # Fix this!
    os.makedirs(dir_path, exist_ok=True)
    for category in category_list_non_repeat:
        os.makedirs(os.path.join(dir_path, category), exist_ok=True)

    # - - - - - prepare models - - - - - # 
    model, autoencoder, text_encoder, diffusion, saved_config = load_ckpt(config.ckpt)

    ### Define audio_encoder (before MLP layer: C=768) -------------------------------------------- ###
        # (Change Line 35 in [config/vggsound_audio.yaml] as in_dim: 512 -> "in_dim: 768" to use this audio encoder)
    audio_encoder = Adapt_CLAP_Module(enable_fusion=False)
    audio_encoder.load_ckpt() # download the default pretrained checkpoint.
    audio_encoder.eval()
    disable_grads(audio_encoder)
    ### ------------------------------------------------------------------ ###

    # ### Define audio_encoder (after MLP layer: C=512) -------------------------------------------- ###
    # audio_encoder = laion_clap.CLAP_Module(enable_fusion=False)
    # audio_encoder.load_ckpt() # download the default pretrained checkpoint.
    # audio_encoder.eval()
    # disable_grads(audio_encoder)
    # ### ------------------------------------------------------------------ ###

    grounding_tokenizer_input = instantiate_from_config(saved_config['grounding_tokenizer_input']) # You can replace saved_config by config
    model.grounding_tokenizer_input = grounding_tokenizer_input
    
    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=config.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 
    
    # - - - - - prepare dataset and dataloader - - - - - # 
    dataset_valid = ConCatDataset(config.val_dataset_names, config.DATA_ROOT, train=False, repeats=None)
    loader_valid = DataLoader(dataset_valid, batch_size=config.batch_size,
                                             shuffle=False, num_workers=config.workers, pin_memory=True, drop_last=False)
    
    # # A list of image file paths and a list of audio file paths
    # image_files = recursively_read_images(rootdir=image_rootdir, must_contain="")
    # image_files.sort()
    # audio_files = recursively_read_audios(rootdir=audio_rootdir, must_contain="")
    # audio_files.sort()
    # - - - - - run iteration - - - - - # 
    tqdm_loader = tqdm(loader_valid)
    for i, batch in enumerate(tqdm_loader): 
        if i == tqdm_loader.total - 1:
            config.batch_size = len(batch['audio_path'])
        # - - - - - prepare input data - - - - - # 
        batch_to_device(batch, device)
        audio_embeddings = get_audio_embeddings(batch, audio_encoder)
        grounding_input = model.grounding_tokenizer_input.prepare(batch, audio_embeddings, no_random_drop=True)
        context = text_encoder.encode(batch["caption"])
        uc = text_encoder.encode( config.batch_size*[""])
        if config.negative_prompt is not None:
            uc = text_encoder.encode(config.batch_size*[config.negative_prompt])

        # - - - - input for gligen - - - - - #
        grounding_extra_input = None
        grounding_downsampler_input = None
        inpainting_extra_input = None
        inpainting_mask = z0 = None

        input = dict(
            x = starting_noise,
            timesteps = None,
            context = context,
            grounding_input = grounding_input,
            inpainting_extra_input = inpainting_extra_input,
            grounding_extra_input = grounding_extra_input,
        )

        # - - - - - start sampling - - - - - #
        shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

        samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
        samples_fake = autoencoder.decode(samples_fake)

        # - - - - - save - - - - - #
        for audio_file, sample in zip(batch["audio_path"], samples_fake):
            for idx in range(len(name_category)):
                if os.path.basename(audio_file).split('.')[0] == os.path.basename(name_category[idx].split(',')[1]).split('.')[0]:
                    category_name = category_list[idx]
                    break
            file_save_path = os.path.join(dir_path, category_name, os.path.basename(audio_file).split('.')[0] + '.jpg')
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1,2,0) * 255 
            sample = Image.fromarray(sample.astype(np.uint8))
            sample.save(file_save_path)



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_ROOT", type=str,  default="/data2/", help="path to DATA")
    parser.add_argument("--folder", type=str,  default="/data2/jungwon/AudScene/generated_images", help="root folder for output")


    parser.add_argument("--name", type=str,  default="test_0", help="experiment will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--yaml_file", type=str,  default="configs/vggsound_audio.yaml", help="paths to base configs.")
    parser.add_argument("--batch_size", type=int, default=5, help="")
    parser.add_argument("--workers", type=int,  default=1, help="")
    parser.add_argument("--ckpt", type=str, default="/data2/jungwon/AudScene/gligen_checkpoints/test_0/tag00/checkpoint_latest.pth")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--csv_path", type=str, default="/data2/VGGSound/vggsound_path.csv", help="original csv class file")
    parser.add_argument("--csv_path2", type=str, default="/data2/jungwon/vggsound_small_with_class.csv", help="small csv class file")

    #parser.add_argument("--negative_prompt", type=str,  default=None, help="")
    args = parser.parse_args()

    config = OmegaConf.load(args.yaml_file)
    config.update( vars(args) )

    run(config)



