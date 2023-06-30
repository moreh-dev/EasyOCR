# Create conda environment and install dependencies
base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
conda create --name easyocr_recognition python=3.8 -y
source ${base_env}/etc/profile.d/conda.sh
conda activate easyocr_recognition 
pip install -r requirements.txt
moreh-switch-model -M 2
echo -e "\\n" | update-moreh --torch 1.7.1 --target 23.6.0 --force

# Train model
python trainer.py --yaml=custom_data_train

# Delete conda environment
echo "deleting env.."
conda deactivate
conda env remove -n easyocr_recognition