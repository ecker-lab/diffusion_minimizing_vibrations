import os, argparse, torch
import time
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchinfo import summary
import numpy as np

from diffusion_plate_optim.pattern_generation import PlateDataset
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator, notebook_launcher
from diffusion_plate_optim.utils.guidance_utils import get_diffusion_model, DiffusionConfig

def get_training_data():
    trainset = PlateDataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=4)

    preprocess = transforms.Compose(
        [   transforms.CenterCrop((73, 113)),
            transforms.Resize((config.image_size[0], config.image_size[1])),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return trainloader, trainset, preprocess


def get_model_and_stuff(trainset):
    model = get_diffusion_model()

    sample_image = trainset[0]['bead_patterns'].unsqueeze(0)
    print(sample_image.shape)
    summary(model, input_data=(preprocess(sample_image), 0))

    noise_scheduler = DDPMScheduler(num_train_timesteps=500)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(trainloader) * config.num_epochs),
    )
    return model, noise_scheduler, optimizer, lr_scheduler


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images
    image_grid = make_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    accelerator.init_trackers("generative_plate")

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        losses = []
        for step, batch in enumerate(train_dataloader):
            clean_images = preprocess(batch['bead_patterns'])
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            losses.append(loss.item())
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        if epoch % 2 == 0:
            print(epoch, f"Loss: {np.mean(losses):4.4f}")

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="debug")
    args = parser.parse_args()

    config = DiffusionConfig()
    save_dir = os.path.join(args.save_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    config.output_dir = save_dir

    trainloader, trainset, preprocess = get_training_data()
    model, noise_scheduler, optimizer, lr_scheduler = get_model_and_stuff(trainset)
    args = (config, model, noise_scheduler, optimizer, trainloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)
