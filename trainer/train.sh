# Create conda environment and install dependencies
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
env_name=easyocr_recognition
batch_szie=32

source ${base_env}/etc/profile.d/conda.sh

if conda env list | grep -q -E "^$env_name\s"; then
    conda activate ${env_name}
else
    conda clean --all --force-pkgs-dir -y
    conda create --name ${env_name} python=3.8 -y
    conda activate ${env_name}
    pip install -r requirements.txt
    moreh-switch-model -M 2
    echo -e "\\n" | update-moreh --torch 1.7.1 --target 23.6.0 --force
fi

# Train model
log_dir="logs_${env_name}"
mkdir -p $log_dir
python trainer.py --batch_size=$batch_szie >> "${log_dir}/log_terminal.log" 2>&1 &
# Capture the process ID (PID) of the training script
pid=$!
bash memory_record_moreh.sh $pid $env_name $batch_szie

# Delete conda environment
echo "deleting env.."
conda deactivate
# conda env remove -n $env_name