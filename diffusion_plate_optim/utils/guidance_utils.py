import os, time
import logging

from dataclasses import dataclass
import torch
import torch.cuda.amp as amp

import numpy as np
from diffusers import UNet2DModel
import matplotlib.pyplot as plt

from diffusion_plate_optim.utils.regression_model_utils import get_mean_from_field_solution, get_net

def pipeline(diffusion_model, scheduler, n_step):
    image = torch.randn((8, 1, 64, 96)).cuda().requires_grad_()
    n_step = 500
    with torch.no_grad():
        diffusion_model.eval()
        for t in np.arange(n_step)[::-1]:
            model_output = diffusion_model(image, t).sample
            current_sample_coeff, pred_original_sample_coeff, pred_original_sample, variance = scheduler.calculate_step(t, model_output, image)

            image = scheduler.step(current_sample_coeff, pred_original_sample_coeff, pred_original_sample, variance, image)
        image = (image / 2 + 0.5).clamp(0, 1)
    return image.detach().cpu().numpy()[:, 0], image


def get_velocity_sum(response):
    vel_sum = response.sum()
    return vel_sum


def get_moments_from_npz(npz_path="moments.npz"):
    data = np.load(npz_path)
    field_mean, field_std = data["field_mean"], data["field_std"]
    out_mean, out_std = data["out_mean"], data["out_std"]
    field_mean, field_std = torch.from_numpy(field_mean), torch.from_numpy(field_std)
    return out_mean, out_std, field_mean, field_std


def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, '
            f'"silent" or None, but got {type(logger)}')


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('acoustics')
    if logger.hasHandlers():
        return logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    if not is_main_process():
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    return logger


def init_train_logger(save_dir, args):
    os.makedirs(os.path.abspath(save_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(save_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{args}')
    return logger, timestamp


@dataclass
class DiffusionConfig:
    image_size = (64, 96)
    train_batch_size = 512
    eval_batch_size = 16
    num_epochs = 250
    gradient_accumulation_steps = 1
    learning_rate = 3e-4
    lr_warmup_steps = 250
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'
    push_to_hub = False
    hub_private_repo = True
    overwrite_output_dir = True
    seed = 0


def get_diffusion_model(config=DiffusionConfig()):
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(16, 32, 64, 64, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
        norm_num_groups=16
    )
    return model


def load_diffusion_model(path):
    diffusion_model = get_diffusion_model()
    diffusion_model.load_state_dict(torch.load(path))
    diffusion_model = diffusion_model.cuda()
    return diffusion_model


def extract_and_process(tensor):
    B, _, H, W = tensor.shape
    border_width = 5
    # Extract borders
    top = tensor[:, :, :border_width, :].reshape(B, -1)
    bottom = tensor[:, :, -border_width:, :].reshape(B, -1)
    left = tensor[:, :, border_width:-border_width, :border_width].reshape(B, -1)
    right = tensor[:, :, border_width:-border_width, -border_width:].reshape(B, -1)
    borders = torch.cat([top, bottom, left, right], dim=1)

    values, _ = torch.kthvalue(borders, int(borders.size(1) * 0.2), dim=1)
    mask = borders <= values.unsqueeze(1)
    selected_values = torch.where(mask, borders, torch.tensor(float('nan')))
    mean_values = torch.nanmean(selected_values, dim=1)
    return mean_values


def get_recon_image(image):
    recon_img = (image + 1) / 2
    recon_img = torch.nn.functional.interpolate(recon_img, (73, 113), align_corners=True, mode="bilinear")
    recon_img =  torch.nn.functional.pad(recon_img, pad=(4, 4, 4, 4), mode='constant', value=0)
    recon_img = recon_img.clamp(0, 1)
    return recon_img


def get_prediction_from_model(image, regression_net, field_mean, field_std):
    recon_img = get_recon_image(image)**2 # the square makes very small values smaller, e.g. closer to zero which mirrors more closely the input data distribution
    prediction_field = regression_net(recon_img)
    prediction = get_mean_from_field_solution(prediction_field, field_mean.cuda(), field_std.cuda())
    return prediction, recon_img


def cosine_scheduler(current_step, total_steps, max_lr, min_lr):
    return 0.5 * (1 + np.cos(np.pi * current_step / total_steps)) * (max_lr - min_lr) + min_lr


def load_regression_model(path):
    net = get_net().cuda()
    net.load_state_dict(torch.load(path)["model_state_dict"])
    return net

def loss_function_wrapper(min_freq, max_freq):
    def loss_function(predictions):
        subset_predictions = predictions[:, min_freq:max_freq]
        return subset_predictions.sum(), len(subset_predictions.flatten())
    return loss_function


def diffusion_guidance(diffusion_net, regression_net, noise_scheduler, field_mean, field_std, loss_fn=None, image=None, n_steps=500, batch_size=1, do_diffusion=True, logger=None, do_range_loss=False, plot=False):
    if loss_fn is not None:
        loss_function = loss_fn
    else:
        if do_range_loss is True:
            loss_function = loss_function_wrapper(100, 200)
        else:
            loss_function = loss_function_wrapper(0, 300)
    scaler = amp.GradScaler(enabled=True)

    if image is None:
        image = torch.randn((batch_size, 1, 64, 96)).cuda().requires_grad_() -0.1
    image_snapshots = []
    regression_net.eval(), diffusion_net.eval()
    for iteration, timestep in enumerate(np.arange(n_steps)[::-1]):
        #do diffusion
        image_old = get_recon_image(image)[0][0].detach().cpu()
        if do_diffusion:
            with torch.no_grad():
                model_output = diffusion_net(image, timestep).sample
                image = noise_scheduler.step(model_output, timestep, image).prev_sample.requires_grad_()

        gradient_diff_norm = torch.linalg.norm(image_old - get_recon_image(image)[0][0].detach().cpu())

        # do optim
        with amp.autocast():  # Enable mixed precision
            prediction, recon_img = get_prediction_from_model(image, regression_net, field_mean, field_std)
            image_old = get_recon_image(image)[0][0].detach().cpu()
            loss, n_sum_elements = loss_function(prediction)

        # update step
        grad = torch.autograd.grad(loss, image)
        image = image - grad[0] * cosine_scheduler(iteration, n_steps, 1e-2, 1e-4)
        recon_img = get_recon_image(image)
        gradient_norm = torch.linalg.norm(image_old - recon_img[0][0].detach().cpu())

        image_snapshots.append((recon_img.detach().cpu().numpy(), prediction.detach().cpu().numpy()))
        if image.grad is not None:
            image.grad.zero_()
        if iteration % 100 == 0 or iteration == n_steps - 1:
            print(f"Diffusion change norm: {gradient_diff_norm:4.3f}, Regression change norm: {gradient_norm:4.3f}")
            print_log(f"Iteration: {iteration}, Loss: {(loss.detach().cpu().numpy()):4.4f}", logger=logger)
            if plot is True:
                plt.imshow(image_snapshots[-1][0][0][0], cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.show()
    return image_snapshots
