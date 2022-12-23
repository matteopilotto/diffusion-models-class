import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from diffusers import DDPMPipeline, DDIMScheduler
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm
from matplotlib import pyplot
from fastcore.script import call_parse
from typing import Dict, List
import random


@call_parse
def train(
    image_size=256,
    batch_size=4,
    num_epochs=1,
    lr=1e-5,
    grad_accumulation_steps=2,
    pretrained_model_name='google/ddpm-bedroom-256',
    dataset_name='huggan/wikiart',
    model_save_name='wikiart_1e',
    wandb_project_name='dm_finetune',
    log_samples_every=50,
    save_model_every=250,
    device='cuda',
    seed=42,
    ):
    
    # start WANDB logging
    wandb.login()
    wandb.init(project=wandb_project_name)
    
    # init DDPM pipeline (for training)
    pipeline = DDPMPipeline.from_pretrained(pretrained_model_name).to(device)
    
    # init sampling schedueler (for inference)
    sampling_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name)
    sampling_scheduler.set_timesteps(num_inference_steps=40)
    
    # load dataset
    dataset = load_dataset(dataset_name, split='train')
    
    # define dataset preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # define function to apply preprocessing pipeline to batches of data (i.e. images)
    def preprocess_fn(examples: Dict[str, List]) -> Dict[str, List]:
        images = [preprocess(image) for image in examples['image']]
        
        return {'images': images}
    
    # apply preprocess to dataset
    dataset.set_transform(preprocess_fn)
    
    # define dataloader
    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # init class object to convert tensors to PIL images
    tensor2pil = transforms.ToPILImage()
    
    # define optimizer and lr scheduler
    model = pipeline.unet
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # training/fine-tuning loop
    # loop over epochs
    
    for epoch in tqdm(range(num_epochs), leave=True, position=0):
        for step, batch in enumerate(tqdm(train_dl, leave=True, position=0)):
            model.train()
            
            images = batch['images'].to(device)
            noise = torch.randn_like(images).to(device)
            
            batch_size = images.shape[0]
            scheduler_timesteps = pipeline.scheduler.num_train_timesteps
            timesteps = torch.tensor([random.randint(0, scheduler_timesteps-1) for _ in range(batch_size)]).to(device)
            
            noisy_images = pipeline.scheduler.add_noise(images, noise, timesteps)
            
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            
            loss = F.mse_loss(noise_pred, noise)
            wandb.log({'loss': loss})
            
            loss.backward()
            
            if (step+1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # log sample images every k steps
            if step == 0 or (step+1) % log_samples_every == 0:
                model.eval()
                generator = torch.Generator(device=device).manual_seed(seed)
                x = torch.randn(8, 3, 256, 256, device=device, generator=generator)
                
                for timestep in tqdm(sampling_scheduler.timesteps, leave=True, position=0):
                    model_input = sampling_scheduler.scale_model_input(x, timestep)
                    
                    with torch.inference_mode():
                        noise_pred = model(model_input, timestep)['sample']
                        x = sampling_scheduler.step(noise_pred, timestep, x, generator=generator).prev_sample
                
                pred_images = x.clip(-1, 1) * 0.5 + 0.5
                wandb.log({'sample images': [wandb.Image(tensor2pil(image)) for image in pred_images]})
                
            if (step+1) % save_model_every == 0:
                pipeline.save_pretrained(save_directory=f'{model_save_name}_{step+1}')
            
        lr_scheduler.step()
    
    pipeline.save_pretrained(save_directory=f'{model_save_name}_final')
    wandb.finish()
                
            
        