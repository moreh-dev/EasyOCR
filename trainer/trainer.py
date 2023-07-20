import os
import torch.backends.cudnn as cudnn
import yaml
from train_mlflow import train
from utils import AttrDict
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="recognition train")
parser.add_argument('--batch_size', type=int)
parser.add_argument('--mlflow_uri', type=str)
args = parser.parse_args()

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
    # Add batch size as an argument
    if args.batch_size!=0 or args.batch_size!=None:
        opt.batch_size = args.batch_size
    if args.mlflow_uri!=None:
        opt.mlflow_uri = args.mlflow_uri
    return opt

opt = get_config("config_files/en_filtered_config.yaml")
os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)

train(opt, amp=False)