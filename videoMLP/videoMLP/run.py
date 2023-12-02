import dataset
import config
from videoMLP.distributed_train import run_distributed_training
from utils.save import save_figs_and_metrics

if __name__ == "__main__":
    with open("config.py", "r") as f:
        print(f.read())
    print("using", config.DEVICE)

    waic_dataset = dataset.get_dataset()

    B_dict = config.B_DICT
    outputs = {}
    for run_name in B_dict:
        print("starting training for", run_name)
        outputs[run_name] = run_distributed_training(run_name, B_dict[run_name], waic_dataset)
        # outputs[run_name]['PEG'] = train_model(network_size, learning_rate, iters, B_dict[run_name], train_data, test_data, True)
        
        save_figs_and_metrics(outputs)

    print("completed execution of run.py")