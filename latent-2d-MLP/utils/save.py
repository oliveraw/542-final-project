import config

import torch
# import torchvision.io as io
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os

def pred2img(pred):
    pred = pred.detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    return pred

def save_final_checkpoint(run_name, dataset, model, B):
    save_image_predictions(run_name, dataset, model)
    save_latents_model_and_B(run_name, model, B)
    create_interpolations(run_name, dataset, model)

def save_image_predictions(run_name, dataset, model):
    output_dir = os.path.join(config.OUTPUT_DIR, run_name, "preds")
    os.makedirs(output_dir, exist_ok=True)

    pe_coords, img, idx = next(iter(dataset))
    pe_coords = pe_coords.to(config.DEVICE)
    print("pe_coords shape", pe_coords.shape)

    with torch.no_grad():
        pred = model(pe_coords, idx)

        print("pred shape", pred.shape)
        batch_size = config.BATCH_SIZE
        for i in range(batch_size):
            pred_path = os.path.join(output_dir, f"{i}.png")
            pred_single = pred[i, :, :, :]
            pred_single = pred2img(pred_single)
            imageio.imwrite(pred_path, pred_single)


def save_latents_model_and_B(run_name, model, B):
    output_dir = os.path.join(config.OUTPUT_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if B != None:
        B_path = os.path.join(output_dir, "B.npy")
        B = B.cpu().numpy()
        np.save(B_path, B)

    model_path = os.path.join(output_dir, "CombinedModel.pth")
    torch.save(model.state_dict(), model_path)


def create_interpolations(run_name, dataset, model):
    save_path = os.path.join(config.OUTPUT_DIR, run_name, f"interpolation.png")

    pe_coords, _, _ = next(iter(dataset))
    pe_coords = pe_coords.to(config.DEVICE)
    with torch.no_grad():
        batch_interpolations, weights = model.interpolate(pe_coords, 0, 1)     # 0, 1 as interpolation indices, returns a list of (# weights, B, 224, 224, 3) imgs
        num_weights, _, _, _, _ = batch_interpolations.shape
        interpolations = batch_interpolations[:, 0, :, :, :]
        fig, axes = plt.subplots(nrows=1, ncols=num_weights, figsize=(10, 3))
        fig.suptitle(f"Interpolations for {run_name}")
        for i in range(num_weights):
            interpolation = interpolations[i]
            weight = weights[i].item()
            axes[i].imshow(pred2img(interpolation))
            axes[i].set_title(weight)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        plt.savefig(save_path)


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
            "Test": {
                run_name: outputs[run_name]['psnrs'] for run_name in outputs
            },
        }
    if config.RECORD_SSIM:
        metrics["SSIM"] = {
            "Test": {
                run_name: outputs[run_name]['ssims'] for run_name in outputs
            },
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