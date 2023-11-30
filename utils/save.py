import config

import os
import torch
import cv2
import numpy as np
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

    videos = {
        "Train": generated_video_train,
        "Test": generated_video_test
    }
    save_videos(output_dir, i, videos)
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
    save_figs(output_dir, i, run_name, record_iterations, metrics)


# videos come in as (T, C, H, W) numpy array
def save_videos(output_dir, i, videos):
    # train_dir_path = os.path.join(output_dir, f"videoMLP_train_{i}")
    # test_dir_path = os.path.join(output_dir, f"videoMLP_test_{i}")
    # os.makedirs(train_dir_path, exist_ok=True)
    # os.makedirs(test_dir_path, exist_ok=True)

    for split, video in videos.items():
        (T, _, H, W) = video.shape
        video_path = os.path.join(output_dir, f"videoMLP_{split}_{i}{config.VIDEO_SUFFIX}")

        # cv2 needs (H, W, 3) for writing each frame
        video = np.transpose(video, (0, 2, 3, 1))

        fourcc = cv2.VideoWriter_fourcc(*config.CODEC)
        save_video_train = cv2.VideoWriter(video_path, fourcc, config.FPS, (H, W))

        print("writing train video of shape", video.shape, video_path)
        for frame in range(T):
            save_video_train.write((video[frame] * 255).astype(np.uint8))   # cv2 needs uint8 in [0, 255] format
            # save_image(generated_video_train[frame], os.path.join(train_dir_path, f"{frame}.png"), format="png")
            # save_image(generated_video_test[frame], os.path.join(test_dir_path, f"{frame}.png"), format="png")

        save_video_train.release()
        cv2.destroyAllWindows()

    
def save_model(output_dir, i, model):
    model_path = os.path.join(output_dir, f"videoMLP_{i}.pth")
    torch.save(model, model_path)


def save_figs(output_dir, i, run_name, record_iterations, metrics):
    for metric in metrics:
        for split in metrics[metric]:
            path = os.path.join(output_dir, f"{metric}_{split}_{i}.png")

            metrics_data = metrics[metric][split]
            print("writing metric", metric, split, metrics_data, record_iterations)
            plt.plot(record_iterations, metrics_data, label=run_name)
            plt.legend(loc='upper left')
            plt.title(split)
            plt.xlabel("Iterations")
            plt.ylabel(metric)
            plt.savefig(path)