import config

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def save_checkpoint(run_name,
                    i, 
                    generated_video_train, 
                    generated_video_test,
                    model):
    output_dir = os.path.join(config.OUTPUT_DIR, f"{i}/{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    videos = {
        "Train": generated_video_train,
        "Test": generated_video_test
    }
    save_videos(output_dir, i, videos)
    save_model(output_dir, i, model)
    # metrics = {}
    # if config.RECORD_PSNR:
    #     metrics["PSNR"] = {
    #         "Train": train_psnrs,
    #         "Test": test_psnrs,
    #     }
    # if config.RECORD_SSIM:
    #     metrics["SSIM"] = {
    #         "Train": train_ssims,
    #         "Test": test_ssims
    #     }
    # save_figs(output_dir, i, run_name, iterations, metrics)


def save_figs_end(outputs):
    # output[run_name] dict
    # {
    #         'iterations': iterations,
    #         'train_psnrs': train_psnrs,
    #         'test_psnrs': test_psnrs,
    #         'train_ssims': train_ssims,
    #         'test_ssims': test_ssims,
    #         'pred_train_vids': np.stack(all_generated_videos_train),
    #         'pred_test_vids': np.stack(all_generated_videos_test),
    # }

    output_dir = os.path.join(config.OUTPUT_DIR, f"final")
    os.makedirs(output_dir, exist_ok=True)

    iterations = range(0, config.ITERATIONS+1, config.RECORD_METRICS_INTERVAL)
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
    save_figs(output_dir, "final", iterations, metrics)


# videos come in as (T, C, H, W) numpy array
def save_videos(output_dir, i, videos):
    for split, video in videos.items():
        video_dir_path = os.path.join(output_dir, f"videoMLP_{split}_{i}")
        os.makedirs(video_dir_path, exist_ok=True)

        (T, _, _, _) = video.shape
        print("writing train video of shape", video.shape, video_dir_path)
        for frame in range(T):
            save_image(video[frame], os.path.join(video_dir_path, f"{frame}.png"), format="png")

    
def save_model(output_dir, i, model):
    model_path = os.path.join(output_dir, f"videoMLP_{i}.pth")
    torch.save(model, model_path)


# def save_figs(output_dir, i, run_name, iterations, metrics):
#     for metric in metrics:
#         for split in metrics[metric]:
#             data_path = os.path.join(output_dir, f"{metric}_{split}_{i}.npy")
#             fig_path = os.path.join(output_dir, f"{metric}_{split}_{i}.png")

#             metrics_data = metrics[metric][split]

#             print("writing metric", metric, split, metrics_data, iterations)
#             np.save(data_path, metrics_data)

#             plt.plot(iterations, metrics_data, label=run_name)
#             plt.legend(loc='upper left')
#             plt.title(split)
#             plt.xlabel("Iterations")
#             plt.ylabel(metric)
#             plt.savefig(fig_path)

def save_figs(output_dir, i, iterations, metrics):
    for metric in metrics:
        for split in metrics[metric]:
            plt.clf()
            fig_path = os.path.join(output_dir, f"{metric}_{split}_{i}.png")
            for run_name in metrics[metric][split]:
                metrics_data = metrics[metric][split][run_name]
                print("writing metric", i, metric, split, run_name, metrics_data)

                data_path = os.path.join(output_dir, f"{metric}_{split}_{i}_{run_name}.npy")
                np.save(data_path, metrics_data)

                plt.plot(iterations, metrics_data, label=run_name)
                plt.legend(loc='upper left')
                plt.title(split)
                plt.xlabel("Iterations")
                plt.ylabel(metric)
                plt.savefig(fig_path)