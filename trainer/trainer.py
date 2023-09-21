import os
import torch.backends.cudnn as cudnn
import yaml
from train_mlflow import train
from utils import AttrDict
import pandas as pd
import argparse

cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    return opt

opt = get_config("trainer/config_files/en_filtered_config.yaml")

parser = argparse.ArgumentParser(description="recognition train")

for key, value in opt.items():
    parser.add_argument(f'--{key}', type=type(value), default=value)
    
parser.add_argument('--config', type=str, default="trainer/config_files/en_filtered_config.yaml")
parser.add_argument('--amp', type=bool, default=False)
args = parser.parse_args()

opt = get_config(args.config)
for key, value in opt.items():
    opt[key] = getattr(args, key)

os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)

train(opt, amp=args.amp)