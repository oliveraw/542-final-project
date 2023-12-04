import numpy as np
import config
from utils import save_figs_and_metrics

import os
import collections

# not enough gpu time to run the full training, had to split into 2 halves:
# wrote numpy files to output directory, load them and generate full figures for metrics

# expected input form to save_figs_and_metrics
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

if __name__ == "__main__":
    output_dir = config.OUTPUT_DIR
    
    outputs = collections.defaultdict(lambda: collections.defaultdict(dict))
    print(os.listdir(output_dir))
    for metric_dir_name in os.listdir(output_dir):
        metric_dir = os.path.join(output_dir, metric_dir_name)
        if os.path.isdir(metric_dir) and not metric_dir_name.isnumeric():
            metrics_filenames = [x for x in os.listdir(metric_dir) if x.endswith(".npy")]
            # print(metric_dir, metrics_filenames)          

            metrics_identifiers = [x.rstrip(".npy") for x in metrics_filenames]
            for idx, metric_identifier in enumerate(metrics_identifiers):
                filename = os.path.join(metric_dir, metrics_filenames[idx])
                metric_data = np.load(filename)
                metric, split, run_name = str.split(metric_identifier, "_")
                if metric == "PSNR":
                    if split == "Train":
                        outputs[run_name]['train_psnrs'] = metric_data
                    else:
                        outputs[run_name]['test_psnrs'] = metric_data
                if metric == "SSIM":
                    if split == "Train":
                        outputs[run_name]['train_ssims'] = metric_data
                    else:
                        outputs[run_name]['test_ssims'] = metric_data

    print(outputs)
    save_figs_and_metrics(outputs)
