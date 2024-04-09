import os, argparse, time, torch
import numpy as np
from acousticnn.plate.configs.main_dir import main_dir
from torchinfo import summary

from diffusers import DDPMScheduler
from diffusion_plate_optim.utils.guidance_utils import diffusion_guidance, load_regression_model, load_diffusion_model, \
    get_moments_from_npz, init_train_logger, get_velocity_sum, print_log


base_path = os.path.join(main_dir, "experiments")
experiment_path = os.path.join(main_dir, "experiments")
min_freq, max_freq = 150, 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_path', type=str)
    parser.add_argument('--regression_path', type=str)
    parser.add_argument('--dir', default="debug", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--n_steps', default=500, type=int)
    parser.add_argument('--do_range_loss', action='store_true', help='specifies loss function for guidance')

    args = parser.parse_args()
    save_dir = os.path.join(args.dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    logger, timestamp = init_train_logger(save_dir, args)
    out_mean, out_std, field_mean, field_std = get_moments_from_npz()
    diffusion_model = load_diffusion_model(args.diffusion_path)
    summary(diffusion_model)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.n_steps)
    regression_model = load_regression_model(args.regression_path)
    image_snapshots = diffusion_guidance(diffusion_model, regression_model, noise_scheduler, field_mean, field_std, loss_fn=None, n_steps=args.n_steps, do_diffusion=True, batch_size=args.batch_size, logger=logger, \
                                do_range_loss=args.do_range_loss)

    # sorting and saving
    images, predictions = image_snapshots[-1][0], image_snapshots[-1][1]
    sort_idx = np.argsort([get_velocity_sum(pred) for pred in predictions])
    images, predictions = images[sort_idx], predictions[sort_idx]
    for i in range(len(images)):
        print_log(f"Sample {i}", logger=logger)
        img, prediction = images[i], predictions[i]
        dictionary = {"image": img, "prediction": prediction}
        torch.save(dictionary, f"{save_dir}/sample_{i}.pt")

    velocity_sum_preds = [get_velocity_sum(pred) for pred in predictions]
    print_log(f"Mean prediction: {np.mean(velocity_sum_preds):5.4f}", logger=logger)
    print_log(f"Std prediction: {np.std(velocity_sum_preds):5.4f}", logger=logger)
    print_log(f"predictions: {velocity_sum_preds}", logger=logger)
