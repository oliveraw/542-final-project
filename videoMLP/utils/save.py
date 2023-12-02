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
                    generated_video_train, 
                    generated_video_test,
                    model):
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, f"{i}/{run_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    videos = {
        "Train": generated_video_train,
        "Test": generated_video_test
    }
    save_videos(checkpoint_dir, i, videos)
    save_model(checkpoint_dir, i, model)


# videos come in as (T, C, H, W) numpy array
def save_videos(checkpoint_dir, i, videos):
    for split, video in videos.items():
        gif_path = os.path.join(checkpoint_dir, f"videoMLP_{split}_{i}.gif")

        print("writing train video of shape", video.shape, gif_path)
        video = (video * 255).astype(np.uint8)
        video = np.transpose(video, (0, 2, 3, 1))
        imageio.mimsave(gif_path, video, format=config.VIDEO_FORMAT, fps=config.FPS)
        

def save_model(checkpoint_dir, i, model):
    model_path = os.path.join(checkpoint_dir, f"videoMLP_{i}.pth")
    torch.save(model, model_path)


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

    iterations = range(0, config.ITERATIONS, config.RECORD_METRICS_INTERVAL)
    metrics = {}
    if config.RECORD_PSNR:
        metrics["PSNR"] = {
            "Train": {
                run_name: outputs[run_name]['train_psnrs'] for run_name in outputs
            },
            "Test": {
                run_name: outputs[run_name]['test_psnrs'] for run_name in outputs
            }
        }
    if config.RECORD_SSIM:
        metrics["SSIM"] = {
            "Train": {
                run_name: outputs[run_name]['train_ssims'] for run_name in outputs
            },
            "Test": {
                run_name: outputs[run_name]['test_ssims'] for run_name in outputs
            }
        }

    for metric in metrics:
        for split in metrics[metric]:
            metric_split_dir = os.path.join(output_dir, f"{metric}_{split}")
            os.makedirs(metric_split_dir, exist_ok=True)

            plt.clf()
            fig_path = os.path.join(metric_split_dir, f"{metric}_{split}.png")
            for run_name in metrics[metric][split]:
                metrics_data = metrics[metric][split][run_name]
                print("writing metric", metric, split, run_name, metrics_data)

                data_path = os.path.join(metric_split_dir, f"{metric}_{split}_{run_name}.npy")
                np.save(data_path, metrics_data)

                plt.plot(iterations, metrics_data, label=run_name)
                plt.legend(loc='upper left')
                plt.title(split)
                plt.xlabel("Iterations")
                plt.ylabel(metric)
                plt.savefig(fig_path)