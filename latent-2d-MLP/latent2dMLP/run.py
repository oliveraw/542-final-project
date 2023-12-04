import config
from latent2dMLP.train import train_model
from utils import save_figs_and_metrics


if __name__ == "__main__":
  with open("config.py", "r") as f:
    print(f.read())

  B_dict = config.B_DICT
  outputs = {}
  for run_name in B_dict:
    print("starting training for", run_name)
    outputs[run_name] = train_model(run_name, B_dict[run_name])

    # do this after training for each run type (in case it gets cut short)
    save_figs_and_metrics(outputs)

  print("completed execution of run.py")