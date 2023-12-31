import os 
import torch as th 



class GroundingNetInput:
    def __init__(self):
        self.set = False 

    def prepare(self, batch, audio_embeddings, no_random_drop=None):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        input only for the ground tokenizer. 

        audio_embeddings: a tensor of shape [batch, num_tokens, C]
        """

        if no_random_drop is None:
            self.set = True # enabale random drop during the training
        else:
            self.set = False # random drop

        mask = batch['mask']
        # bbox = batch['bbox']

        self.batch, self.N, self.C = audio_embeddings.shape
        self.device = audio_embeddings.device
        self.dtype = audio_embeddings.dtype
        # bbox = bbox.reshape(self.batch, 1, 4)
        # bbox = bbox.repeat(1, self.N, 1) # torch.Size([batch, num_tokens, 4])

        return {"audio_embeddings": audio_embeddings, "mask": mask}




    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        audio_embeddings = th.zeros(self.batch, self.N, self.C).type(dtype).to(device)
        mask = th.zeros(batch).type(dtype).to(device)
        # bbox = (th.tensor([0, 0, 511, 511])).repeat(self.batch, self.N, 1).type(dtype).to(device)


        return {"audio_embeddings": audio_embeddings, "mask": mask}









