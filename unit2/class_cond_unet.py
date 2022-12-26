import torch
from torch import nn
import diffusers
from diffusers import UNet2DModel

class ClassConditionedUNet(nn.Module):
    def __init__(self, num_classes=10, class_embed_size=4):
        super().__init__()
        
        # init emebedding layer
        self.embedding_layer = nn.Embedding(num_classes, class_embed_size)
        
        # init model
        self.model = UNet2DModel(
            sample_size=28,
            in_channels=class_embed_size + 1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 64),
            down_block_types=('DownBlock2D',
                              'AttnDownBlock2D',
                              'AttnDownBlock2D'),
            up_block_types=('AttnUpBlock2D',
                            'AttnUpBlock2D',
                            'UpBlock2D')
        )
        
    
    def forward(self, x, timestep, labels):
        # batch size, channels, height, width
        batch_size, ch, h, w = x.shape
        
        # compute class embeddings
        class_embeds = self.embedding_layer(labels)
        
        # reshape class embeddings
        class_embeds = class_embeds.view(*class_embeds.shape, 1, 1).expand(*class_embeds.shape, h, w)
        
        # concatenate input "x" and class embeddings together across channel dimension
        model_input = torch.cat((x, class_embeds), dim=1)
        
        # forward pass
        noise_pred = self.model(model_input, timestep)
        
        return noise_pred