from tkinter.messagebox import NO
import torch 
import json 
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms
from io import BytesIO
import random
import torchvision.transforms.functional as TF

from .tsv import TSVFile

from io import BytesIO
import base64
import numpy as np
import os

import librosa
import laion_clap

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def audio_embed(audio_path):
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() # download the default pretrained checkpoint.

    # Directly get audio embeddings from audio files
    audio_file = os.listdir(audio_path)
    #[
    #    '/home/data/test_clap_short.wav',
    #    '/home/data/test_clap_long.wav'
    #]
    audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
    #print(audio_embed[:,-20:])
    #print(audio_embed.shape)

    return audio_embed

def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr


def decode_item(item):
    "This is for decoding TSV for box data"
    item = json.loads(item)
    item['image'] = decode_base64_to_pillow(item['image'])

    for anno in item['annos']:
        anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
        anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
        anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
    return item


def decode_item_audio(item):
    "This is for decoding TSV for audio data"
    item = json.loads(item)
    item['audio'] = decode_base64_to_pillow(item['audio'])
    return item


class AudioDataset():
    def __init__(self, tsv_path, audio_tsv_path, prob_use_caption=1, image_size=512, random_flip=False):

        self.tsv_path = tsv_path
        self.audio_tsv_path = audio_tsv_path
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.random_flip = random_flip

        # Load tsv data
        self.tsv_file = TSVFile(self.tsv_path)
        self.audio_tsv_file = TSVFile(self.audio_tsv_path)      

        self.pil_to_tensor = transforms.PILToTensor()


    def total_images(self):
        return len(self)


    def get_item_from_tsv(self, index):
        _, item = self.tsv_file[index]
        item = decode_item(item)
        return item


    def get_item_from_audio_tsv(self, index):
        _, item = self.audio_tsv_file[index]
        item = decode_item_audio(item)
        return item



    def __getitem__(self, index):

        raw_item = self.get_item_from_tsv(index)
        raw_item_audio = self.get_item_from_audio_tsv(index)

        assert raw_item['data_id'] == raw_item_audio['data_id']
        
        out = {}

        out['id'] = raw_item['data_id']
        image = raw_item['image']
        audio = raw_item_audio['audio']

        # - - - - - center_crop, resize and random_flip - - - - - - #  
        assert  image.size == audio.size   

        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize( (self.image_size, self.image_size) )

        audio = TF.center_crop(audio, crop_size)
        audio = audio.resize( (self.image_size, self.image_size) )


        if self.random_flip and random.random()<0.5:
            image = ImageOps.mirror(image)
            audio = ImageOps.mirror(audio)
        
        out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
        out['audio'] = ( self.pil_to_tensor(audio).float()/255 - 0.5 ) / 0.5
        out['mask'] = torch.tensor(1.0) 

        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            out["caption"] = raw_item["caption"]
        else:
            out["caption"] = ""

        return out


    def __len__(self):
        return len(self.tsv_file)


