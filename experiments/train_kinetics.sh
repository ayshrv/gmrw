# setting="CUDA_VISIBLE_DEVICES=0 python "
setting="OMP_NUM_THREADS=4 torchrun --nproc_per_node 4 --nnodes 1 "
file="train_kinetics.py"
arguments="
--data_path datasets/kinetics
--no_of_frames 2
--lr 4e-5
--img_size1_h 640
--img_size1_w 640
--img_size2_h 512
--img_size2_w 512
--data_aug_setting setting2
--per_gpu_train_batch_size 1
--per_gpu_eval_batch_size 1
--max_viz_per_batch 1
--train_viz_log_freq 500
--epochs 5000
--visualize
--exp_name train_kinetics
"
command_to_run="${setting} ${file} ${arguments}"

source ~/.bashrc
conda activate long-range-crw_pytorch2.0
echo $command_to_run
echo
eval $command_to_run