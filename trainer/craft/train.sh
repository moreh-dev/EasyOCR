# Create conda environment and install dependencies
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
env_name=easyocr_detection
batch_size=5

source ${base_env}/etc/profile.d/conda.sh

if conda env list | grep -q -E "^$env_name\s"; then
    conda activate ${env_name}
else
    conda clean --all --force-pkgs-dir -y
    conda create --name ${env_name} python=3.8 -y
    conda activate ${env_name}
    pip install -r requirements.txt
    moreh-switch-model -M 2
    echo -e "\\n" | update-moreh --torch 1.13.1 --target 23.6.0 --force
fi
# Download pretrained model
model_dir="pretrained_model/"
mkdir -p "${model_dir}"
pretrained_model_file="${model_dir}/CRAFT_clr_amp_29500.pth"
if [ -f "${pretrained_model_file}" ]; then
    echo "File already exists in ${model_dir}. Skipping download."
else
    echo "File does not exist in ${model_dir}. Downloading..."
    pip install gdown
    gdown --id 1enVIsgNvBf3YiRsVkxodspOn55PIK-LJ --output "${pretrained_model_file}"
fi

# Train model
log_dir="logs_${env_name}"
mkdir -p $log_dir
CUDA_VISIBLE_DEVICES=0 python3 train_mlflow.py --yaml=custom_data_train --batch_size=$batch_size >> "${log_dir}/log_terminal.log" 2>&1 &
# Capture the process ID (PID) of the training script
pid=$!
bash ../memory_record_moreh.sh $pid $env_name $batch_szie

# Delete conda environment
echo "deleting env.."
conda deactivate
# conda env remove -n $env_name