import config

import os
import torch
# from torchvision.io import write_video
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def save_checkpoint(run_name,
                    i, 
                    generated_video_train, 
                    generated_video_test,
                    model,
                    record_iterations, 
                    train_psnrs,
                    test_psnrs, 
                    train_ssims,
                    test_ssims):
    output_dir = os.path.join(config.OUTPUT_DIR, f"{i}/{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    save_videos(output_dir, i, generated_video_train, generated_video_test)
    save_model(output_dir, i, model)
    metrics = {}
    if config.RECORD_PSNR:
        metrics["PSNR"] = {
            "Train": train_psnrs,
            "Test": test_psnrs,
        }
    if config.RECORD_SSIM:
        metrics["SSIM"] = {
            "Train": train_ssims,
            "Test": test_ssims
        }
    save_figs(output_dir, i, record_iterations, metrics)


# videos come in as (T, C, H, W) format
def save_videos(output_dir, i, generated_video_train, generated_video_test):
    train_dir_path = os.path.join(output_dir, f"videoMLP_train_{i}")
    test_dir_path = os.path.join(output_dir, f"videoMLP_test_{i}")
    os.makedirs(train_dir_path, exist_ok=True)
    os.makedirs(test_dir_path, exist_ok=True)

    # generated_video_train = generated_video_train.permute((0, 2, 3, 1))
    # generated_video_test = generated_video_test.permute((0, 2, 3, 1))

    print("writing train video of shape", generated_video_train.shape, train_dir_path)
    t = generated_video_train.shape[0]
    for frame in range(t):
        # print(generated_video_train[frame])
        save_image(generated_video_train[frame], os.path.join(train_dir_path, f"{frame}.png"), format="png")
        save_image(generated_video_test[frame], os.path.join(test_dir_path, f"{frame}.png"), format="png")

    # write_video(train_dir_path, generated_video_train, fps=config.FPS)
    # write_video(test_dir_path, generated_video_test, fps=config.FPS)

    
def save_model(output_dir, i, model):
    model_path = os.path.join(output_dir, f"videoMLP_{i}.pth")
    torch.save(model, model_path)


def save_figs(output_dir, i, record_iterations, metrics):
    for metric in metrics:
        for split in metrics[metric]:
            path = os.path.join(output_dir, f"{metric}_{split}_{i}.png")

            metrics_data = metrics[metric][split]
            print("writing metric", metric, split, metrics_data, record_iterations)
            plt.plot(record_iterations, metrics_data)
            plt.title(split)
            plt.xlabel("Iterations")
            plt.ylabel(metric)
            plt.savefig(path)