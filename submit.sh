#!/bin/bash
#SBATCH --job-name=train_neural-lam-gpu
#SBATCH --output=res_GPU.txt
#SBATCH --partition=gpu
#SBATCH --time=12:00:00          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G                
#SBATCH --gres=gpu:1               
#SBATCH --constraint=v100       



export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



srun pdm run python -m neural_lam.train_model  --config_path scripts/cosmo_model_config_era5.yaml  --model hi_lam  --graph_name rectangular_hierarchical  --hidden_dim 120  --hidden_dim_grid 60  --time_delta_enc_dim 32  --processor_layers 2  --batch_size 1  --min_lr 0.001  --epochs 1  --val_interval 1  --val_steps_to_log 1 6 12 18 24  --ar_steps_eval 24  --precision bf16-mixed  --plot_vars "T_2M"  --num_workers 8  --num_nodes 1
