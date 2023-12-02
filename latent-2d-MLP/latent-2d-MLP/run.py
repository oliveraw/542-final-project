import dataset
import config
from videoMLP.train import train_model
from utils import save_figs_and_metrics

waic_dataset = dataset.get_dataset()

B_dict = config.B_DICT
outputs = {}
for run_name in B_dict:
  print("starting training for", run_name)
  outputs[run_name] = train_model(run_name, B_dict[run_name], waic_dataset)
  # outputs[run_name]['PEG'] = train_model(network_size, learning_rate, iters, B_dict[run_name], train_data, test_data, True)

  # do this after training for each run type (in case it gets cut short)
  save_figs_and_metrics(outputs)

print("completed execution of run.py")