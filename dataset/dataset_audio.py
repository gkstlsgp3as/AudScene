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

### Please check the comments below ------------------------------------------- ###
def recursively_read_images(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    """
    Your dataset must be organized as [rootdir - subdirectories - image files].
        If your dataset is organized as [rootdir - image files], please use 
        "dataset/dataset_sem.py/recursively_read" instead.
    
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
### ---------------------------------------------------------------------- ###

### Please check the comments below ------------------------------------------- ###
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
### ---------------------------------------------------------------------- ###



class AudioDataset():
    def __init__(self, image_rootdir, audio_rootdir, caption_path, prob_use_caption=1, image_size=512, random_flip=False):
        self.image_rootdir = image_rootdir
        self.audio_rootdir = audio_rootdir
        self.caption_path = caption_path
        self.prob_use_caption = prob_use_caption 
        self.image_size = image_size
        self.random_flip = random_flip

        # Image and files
        image_files = recursively_read_images(rootdir=image_rootdir, must_contain="")
        image_files.sort()
        audio_files = recursively_read_audios(rootdir=audio_rootdir, must_contain="")
        audio_files.sort()

        self.image_files = image_files
        self.audio_files = audio_files

        # Open caption json
        with open(caption_path, 'r') as f:
            self.image_filename_to_caption_mapping = json.load(f)
        
        assert len(self.image_files) == len(self.audio_files) == len(self.image_filename_to_caption_mapping)
        self.pil_to_tensor = transforms.PILToTensor()

    def total_images(self):
        return len(self)
    
    def __getitem__(self, index):
        image_path = self.image_files[index]

        out = {}

        out['id'] = index
        out['audio_path'] = str(self.audio_files[index]) # DataLoader's default collate function doesn't know how to handle pathlib.PosixPath objects -> use str()
        image = Image.open(image_path).convert("RGB")

        # - - - - - center_crop, resize and random_flip - - - - - - #  

        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize( (self.image_size, self.image_size) )
        
        if self.random_flip and random.random()<0.5:
            image = ImageOps.mirror(image)
        
        out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
        out['mask'] = torch.tensor(1.0)

        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            out["caption"] = self.image_filename_to_caption_mapping[ os.path.basename(image_path) ]
        else:
            out["caption"] = ""
        
        return out
    
    def __len__(self):
        return len(self.image_files)



# # quantization
# def int16_to_float32(x):
#     return (x / 32767.0).astype(np.float32)


# def float32_to_int16(x):
#     x = np.clip(x, a_min=-1., a_max=1.)
#     return (x * 32767.).astype(np.int16)


# def decode_base64_to_pillow(image_b64):
#     return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

# def decode_tensor_from_string(arr_str, use_tensor=True):
#     arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
#     if use_tensor:
#         arr = torch.from_numpy(arr)
#     return arr


# ### Modify this function ---------------------------------------------- ###
# def audio_embed(audio_path):
#     """
#     This function is used to extract audio embeddings from audio files.
#     Output shape: [num_tokens, hidden_dim]
#     """
#     model = laion_clap.CLAP_Module(enable_fusion=False)
#     model.load_ckpt() # download the default pretrained checkpoint.

#     # Directly get audio embeddings from audio files
#     audio_file = os.listdir(audio_path)
#     #[
#     #    '/home/data/test_clap_short.wav',
#     #    '/home/data/test_clap_long.wav'
#     #]
#     audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
#     #print(audio_embed[:,-20:])
#     #print(audio_embed.shape)

#     return audio_embed
# ### ---------------------------------------------------------------------- ###


# class AudioDataset():
#     def __init__(self, tsv_path, audio_tsv_path, prob_use_caption=1, image_size=512, random_flip=False):

#         self.tsv_path = tsv_path
#         self.audio_tsv_path = audio_tsv_path
#         self.prob_use_caption = prob_use_caption
#         self.image_size = image_size
#         self.random_flip = random_flip

#         # Load tsv data
#         self.tsv_file = TSVFile(self.tsv_path)
#         self.audio_tsv_file = TSVFile(self.audio_tsv_path)      

#         self.pil_to_tensor = transforms.PILToTensor()


#     def total_images(self):
#         return len(self)


#     def get_item_from_tsv(self, index):
#         _, item = self.tsv_file[index]
#         item = decode_item(item)
#         return item


#     def get_item_from_audio_tsv(self, index):
#         _, item = self.audio_tsv_file[index]
#         item = decode_item_audio(item)
#         return item



#     def __getitem__(self, index):

#         raw_item = self.get_item_from_tsv(index)
#         raw_item_audio = self.get_item_from_audio_tsv(index)

#         assert raw_item['data_id'] == raw_item_audio['data_id']
        
#         out = {}

#         out['id'] = raw_item['data_id']
#         image = raw_item['image']
#         audio = raw_item_audio['audio']

#         # - - - - - center_crop, resize and random_flip - - - - - - #  
#         assert  image.size == audio.size   

#         crop_size = min(image.size)
#         image = TF.center_crop(image, crop_size)
#         image = image.resize( (self.image_size, self.image_size) )

#         audio = TF.center_crop(audio, crop_size)
#         audio = audio.resize( (self.image_size, self.image_size) )


#         if self.random_flip and random.random()<0.5:
#             image = ImageOps.mirror(image)
#             audio = ImageOps.mirror(audio)
        
#         out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
#         out['audio'] = ( self.pil_to_tensor(audio).float()/255 - 0.5 ) / 0.5
#         out['mask'] = torch.tensor(1.0) 

#         # -------------------- caption ------------------- # 
#         if random.uniform(0, 1) < self.prob_use_caption:
#             out["caption"] = raw_item["caption"]
#         else:
#             out["caption"] = ""

#         return out


#     def __len__(self):
#         return len(self.tsv_file)


# def decode_item(item):
#     "This is for decoding TSV for box data"
#     item = json.loads(item)
#     item['image'] = decode_base64_to_pillow(item['image'])

#     for anno in item['annos']:
#         anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
#         anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
#         anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
#         anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
#     return item