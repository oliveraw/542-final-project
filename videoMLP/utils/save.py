import config

import os
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# videos and model are saved every config.RECORD_STATE_INTERVAL, metrics and figures are saved at the end of each training run
def save_checkpoint(run_name,
                    i, 
                    generated_videos,
                    model):
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, f"{i}/{run_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    save_videos(checkpoint_dir, generated_videos)
    save_model(checkpoint_dir, model)


# videos come in as (T, C, H, W) numpy array
def save_videos(checkpoint_dir, videos):
    video_dir = os.path.join(checkpoint_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    for idx, video in enumerate(videos):
        gif_path = os.path.join(video_dir, f"{config.VIDEO_NAMES[idx]}.gif")
        print("writing video of shape", video.shape, gif_path)
        video = (255 * video).astype(np.uint8)
        imageio.mimsave(gif_path, video, format=config.VIDEO_FORMAT, fps=config.FPS)


def save_interpolated_videos(run_name, model, first_xyt):
    interpolated_video_dir = os.path.join(config.OUTPUT_DIR, f"interpolations/{run_name}")
    os.makedirs(interpolated_video_dir, exist_ok=True)

    with torch.no_grad():
        interpolations, weights = model.interpolate(first_xyt, 0, 1)     # 0, 1 as interpolation indices, returns a list of (224, 224, 3) imgs
        for i in range(len(interpolations)):
            interpolation = interpolations[i].cpu().detach().numpy()
            weight = weights[i].cpu().item()
            interpolated_video_path = os.path.join(interpolated_video_dir, f"{weight}.gif")
            interpolation = (255 * interpolation).astype(np.uint8)
            imageio.mimsave(interpolated_video_path, interpolation, format=config.VIDEO_FORMAT, fps=config.FPS)
        

def save_model(checkpoint_dir, model):
    model_path = os.path.join(checkpoint_dir, f"videoMLP.pth")
    torch.save(model.state_dict(), model_path)


def save_figs_and_metrics(outputs):
    # outputs[run_name] dict:
    # {
    #         'iterations': iterations,
    #         'train_psnrs': train_psnrs,
    #         'test_psnrs': test_psnrs,
    #         'train_ssims': train_ssims,
    #         'test_ssims': test_ssims,
    #         'pred_train_vids': np.stack(all_generated_videos_train),
    #         'pred_test_vids': np.stack(all_generated_videos_test),
    # }

    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    metrics = {}
    if config.RECORD_PSNR:
        metrics["PSNR"] = {
            "Train": {
                run_name: outputs[run_name]['train_psnrs'] for run_name in outputs
            },
            # "Test": {
            #     run_name: outputs[run_name]['test_psnrs'] for run_name in outputs
            # }
        }
    if config.RECORD_SSIM:
        metrics["SSIM"] = {
            "Train": {
                run_name: outputs[run_name]['train_ssims'] for run_name in outputs
            },
            # "Test": {
            #     run_name: outputs[run_name]['test_ssims'] for run_name in outputs
            # }
        }

    for metric in metrics:
        for split in metrics[metric]:
            metric_split_dir = os.path.join(output_dir, f"{metric}_{split}")
            os.makedirs(metric_split_dir, exist_ok=True)

            plt.clf()
            fig_path = os.path.join(metric_split_dir, f"{metric}_{split}.png")
            for run_name in metrics[metric][split]:
                metrics_data = metrics[metric][split][run_name]
                iterations = np.linspace(0, config.ITERATIONS + 1, len(metrics_data))
                print("writing metric", metric, split, run_name, metrics_data)

                data_path = os.path.join(metric_split_dir, f"{metric}_{split}_{run_name}.npy")
                np.save(data_path, metrics_data)

                plt.plot(iterations, metrics_data, label=run_name)
                plt.legend(loc='upper left')
                plt.title(split)
                plt.xlabel("Iterations")
                plt.ylabel(metric)
                plt.savefig(fig_path)