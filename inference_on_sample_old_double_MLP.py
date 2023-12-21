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
    config['model']['params']['grounding_tokenizer']['target'] = 'ldm.modules.diffusionmodules.audio_grounding_net4_old.PositionNet'
    config['grounding_tokenizer_input']['target'] = 'grounding_input.audio_grounding_tokinzer_input_old.GroundingNetInput'

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
    batch_audio_path = batch['audio_path'] # A list of audio_paths: shape = [B]

    ### Your code here ---------------------------------------- ###
    # audio_embeddings = audio_encoder.get_audio_embedding_from_filelist(x = batch_audio_path, use_tensor=True)
    audio_embeddings = audio_encoder.get_audio_embedding_from_filelist(x = batch_audio_path, use_tensor=True, num_tokens=32)
    if len(audio_embeddings.shape) == 2:
        audio_embeddings = audio_embeddings.unsqueeze(1)  # [B, N (=1), C]
    ### -------------------------------------------------------- ###

    return audio_embeddings

@torch.no_grad()
def prepare_batch(meta, batch_size=1):
    """
    This function is used to prepare a batch for inference.
    batch["audio_path"]: a list of an audio path
    batch["mask"]: torch.ones(batch_size)
    """
    batch = {}
    batch["audio_path"] = [meta["audio_path"]] * batch_size
    batch["mask"] = torch.ones(batch_size)
    # batch["bbox"] = torch.tensor(meta["bbox"]).repeat(batch_size, 1) # torch.Size([batch_size, 4])

    return batch_to_device(batch, device)

@torch.no_grad()
def run(meta, args, starting_noise=None, audio_encoder=None):
    # - - - - - prepare models - - - - - # 
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(args.ckpt)
    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input']) 
    model.grounding_tokenizer_input = grounding_tokenizer_input

    # - - - - - update config from args - - - - - # 
    config.update( vars(args) )
    config = OmegaConf.create(config)

    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=config.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 
    
    # - - - - - prepare batch - - - - - # 
    batch = prepare_batch(meta, args.batch_size)
    context = text_encoder.encode(  [meta["prompt"]]*config.batch_size  )
    uc = text_encoder.encode( config.batch_size*[""] )
    if args.negative_prompt is not None:
        uc = text_encoder.encode( config.batch_size*[args.negative_prompt] )

    # - - - - - input for gligen - - - - - #
    audio_embeddings = get_audio_embeddings(batch, audio_encoder)
    grounding_input = model.grounding_tokenizer_input.prepare(batch, audio_embeddings, no_random_drop=True)

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

    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)

    # - - - - - save - - - - - #
    output_folder = os.path.join(args.folder, meta["save_folder_name"])
    os.makedirs(output_folder, exist_ok=True)

    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.batch_size))
    print(image_ids)
    for image_id, sample in zip(image_ids, samples_fake):
        img_name = f"{image_id}_{meta['class_name']}.png"
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        sample.save(  os.path.join(output_folder, img_name)   )






if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_ROOT", type=str,  default="/data2/", help="path to DATA")
    parser.add_argument("--folder", type=str,  default="/data2/jungwon/AudScene/generated_images", help="root folder for output")

    parser.add_argument("--yaml_file", type=str,  default="jungwon/AudScene/gligen_checkpoints/test_10/tag00/train_config_file.yaml", help="paths to base configs.")
    parser.add_argument("--batch_size", type=int, default=10, help="")
    parser.add_argument("--workers", type=int,  default=1, help="")
    parser.add_argument("--ckpt", type=str, default="/data2/jungwon/AudScene/gligen_checkpoints/test_10/tag00/checkpoint_latest.pth")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    parser.add_argument("--save_folder_name", type=str,  default="try_0_no_prompt", help="")

    args = parser.parse_args()

    alpha_type_ = [1, 0, 0] #[0.9, 0.05, 0.05]
    save_folder_name_ = args.save_folder_name
    # bbox_for_all = [[0, 0, 511, 511]] # [xmin, ymin, xmax, ymax]
    meta_list = [
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/-/-S-TDT5oq0Q_000290.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "cattle mooing"
        ),
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/-/-gSfPQqi6nI_000030.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "cat meowing"
        ),
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/0/0N3-lCzOQPI_000000.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "frog croaking"
        ),
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/1/1JfwuwHV0hc_000212.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "volcano explosion"
        ),
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/2/2XFrBXTnNY8_000080.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "dog barking"
        ),
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/3/3iQ_xRurgS8_000030.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "horse neighing"
        ),
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/9/9AUSYLKYKGg_001113.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "owl hooting"
        ),
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/4/4GGUH7ykdCg_000022.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "duck quacking"
        ),    
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/w/wZr8jXo1Uso_000102.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "hail"
        ),    
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/r/rvNKfBj-Nnk_000030.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "pig oinking"
        ),
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/d/de8skUrbdUc_000030.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "cricket chirping"
        ),              
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/U/UsTk4M-U7-4_000246.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "fire crackling"
        ),    
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/test/g/g5FVJveyyVM_000030.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "lions roaring"
        ),    
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/train/7/7CaBMXaxINY_000070.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "ocean burbling"
        ),   
        dict(
            prompt = "",
            alpha_type = alpha_type_, # If you don't want to use guiding token, set this value to [0, 0, 1]
            audio_path = "/data2/VGGSound/audio_mag20_aud40_1207/train/-/-HMmNXcZe1I_000053.wav",  
            # bbox = bbox_for_all,
            save_folder_name = save_folder_name_,
            class_name = "bird chirping, tweeting"
        ),   
    ]

    ## Define audio_encoder (before MLP layer: C=768) -------------------------------------------- ###
    # (Change Line 35 in [config/vggsound_audio.yaml] as in_dim: 512 -> "in_dim: 768" to use this audio encoder)
    audio_encoder = Adapt_CLAP_Module(enable_fusion=False)
    audio_checkpoint = torch.load("/data2/jungwon/AudScene/contrastive_learned_CLAP6/checkpoint_epoch_30.pth") # /data2/jungwon/AudScene/contrastive_learned_CLAP2/checkpoint_epoch_40.pth
    audio_encoder.load_state_dict(audio_checkpoint['model_state_dict'])
    audio_encoder.eval()
    disable_grads(audio_encoder)
    ## ------------------------------------------------------------------ ###
    # ### Define audio_encoder (before MLP layer: C=768) -------------------------------------------- ###
    # # (Change Line 35 in [config/vggsound_audio.yaml] as in_dim: 512 -> "in_dim: 768" to use this audio encoder)
    # audio_encoder = Adapt_CLAP_Module(enable_fusion=False)
    # audio_encoder.load_ckpt() # download the default pretrained checkpoint.
    # audio_encoder.eval()
    # disable_grads(audio_encoder)
    # ### ------------------------------------------------------------------ ###
    # ### Define audio_encoder (after MLP layer: C=512) -------------------------------------------- ###
    # audio_encoder = laion_clap.CLAP_Module(enable_fusion=False)
    # # audio_checkpoint = torch.load("/data2/jungwon/AudScene/contrastive_learned_CLAP4/checkpoint_epoch_135.pth")
    # # audio_encoder.load_state_dict(audio_checkpoint['model_state_dict'])
    # audio_encoder.load_ckpt()
    # audio_encoder.eval()
    # disable_grads(audio_encoder)
    # ### ------------------------------------------------------------------ ###
    for meta in meta_list:
        run(meta, args, starting_noise=None, audio_encoder=audio_encoder)