# Enhancing Controllability in Audio-to-Scene Generation with a Text-Guided Diffusion Model
 (SNU M3309.002900. Machine Listening) 

[Jungwon Park](https://github.com/Jungwona), [Jungmin Ko](https://github.com/JungminKo), [Shinhye Han](https://gkstlsgp3as.github.io/)

### Abstract
---
Audio-to-scene generation faces challenges in achieving high controllability due to the limited expressiveness of non-verbal audio and a scarcity of high-quality audio-scene datasets. To address this, we refine a widely used video dataset and incorporate text and bounding box guidance into the audio-to-scene generation using an adapted version of GLIGEN. Our method enhances controllability by utilizing the expressive nature of text guidance, correcting errors of a large pre-trained text-to-image generative model through the integration of audio signals. Notably, our approach achieves a state-of-the-art CLIP retrieval score when audio signals are the sole input. These results highlight a promising architectural choice for text-guided audio-to-scene generation. For future work, we aim to refine bounding box labels in the training set to align more accurately with bounding box guidance.

Our model incorporates audio and bounding box information through guiding tokens in GLIGEN, a highly controllable text-to-image generative model. We trained this model with image and audio pairs extracted from video files, coupled with bounding box data obtained from Grounding DINO, an object detector. For generating caption tokens, we employed LLaVA \cite{liu2023visual}, a captioning VLM.


![1_model_architecture](https://github.com/user-attachments/assets/b469906b-7e65-4fb0-b7ed-cd63256cce50)



## Download pretrained GLIGEN models

We provide ten checkpoints for different use scenarios. All models here are based on SD-V-1.4.
| Mode       | Modality       | Download                                                                                                       |
|------------|----------------|----------------------------------------------------------------------------------------------------------------|
| Generation | Box+Text       | [HF Hub](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin)       |
| Generation | Box+Text+Image | [HF Hub](https://huggingface.co/gligen/gligen-generation-text-image-box/blob/main/diffusion_pytorch_model.bin) |
| Generation | Keypoint       | [HF Hub](https://huggingface.co/gligen/gligen-generation-keypoint/blob/main/diffusion_pytorch_model.bin)       |
| Inpainting | Box+Text       | [HF Hub](https://huggingface.co/gligen/gligen-inpainting-text-box/blob/main/diffusion_pytorch_model.bin)       |
| Inpainting | Box+Text+Image | [HF Hub](https://huggingface.co/gligen/gligen-inpainting-text-image-box/blob/main/diffusion_pytorch_model.bin) |
| Generation | Hed map        | [HF Hub](https://huggingface.co/gligen/gligen-generation-hed/blob/main/diffusion_pytorch_model.bin)      |
| Generation | Canny map      | [HF Hub](https://huggingface.co/gligen/gligen-generation-canny/blob/main/diffusion_pytorch_model.bin)      |
| Generation | Depth map      | [HF Hub](https://huggingface.co/gligen/gligen-generation-depth/blob/main/diffusion_pytorch_model.bin)      |
| Generation | Semantic map   | [HF Hub](https://huggingface.co/gligen/gligen-generation-sem/blob/main/diffusion_pytorch_model.bin)      |
| Generation | Normal map     | [HF Hub](https://huggingface.co/gligen/gligen-generation-normal/blob/main/diffusion_pytorch_model.bin)      |

Note that the provided checkpoint for semantic map is only trained on ADE20K dataset; the checkpoint for normal map is only trained on DIODE dataset.

### :point_right: Results
#### Image generation with incremental audio guidance. 
The number of audio-guided sample steps increases by 10\% from left to right. For example, the third column displays images generated with an initial 20\% sample steps guided by both text and audio, followed by the remaining 80\% text-only guided sample steps. The growing influence of audio guidance is evident. The same random seed is used for each row, and all images are generated conditioned on the text prompt ``A photo of the [class name].''

![2_results](https://github.com/user-attachments/assets/ab648c1d-f285-4342-ac07-9e75d23d8aac)

#### Image generation with audio and text guidance.
Upper images illustrate the blended background guided by audio (hail) and text (Times Square or grassland). Lower ones display objects guided by text within the audio-guided background (hail). Our approach proficiently generates images with audio-text bimodal guidance.

![3_ablation](https://github.com/user-attachments/assets/0838c671-2e8b-49ce-8687-08c208fdb651)



### :airplane: Training
```
python main.py --name=depth_test  --yaml_file=./configs/vggsound_audio_X.yaml
```

### :rocket: Inference 
Reproduce the results for 4D Imaging Radar Super-Resolution:
```
# LLaVA Inference
python captioning/llava/eval/model.py --dataset {dataset_name}

# Image Generation
python inference_on_*.py
```

