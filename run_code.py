import subprocess

# Activate the Conda environment
conda_env = 'torch'  # Replace with your Conda environment name
activate_cmd = f'conda activate {conda_env}'
# subprocess.run(activate_cmd, shell=True, check=True)

# DONE:
# '''
#     'python main_pretrain.py --training-fold 2 --gpu-index 2 --dataset first --cls 3 >2.out',
#     'python main_pretrain.py --training-fold 3 --gpu-index 3 --dataset first --cls 3 >3.out',
#     'python main_pretrain.py --training-fold 4 --gpu-index 4 --dataset first --cls 3 >4.out',
#     'python main_pretrain.py --training-fold 5 --gpu-index 5 --dataset first --cls 3 >5.out'
#
#
# '''

# NOT YET:
#'python main_pretrain.py --training-fold 0 --gpu-index 4 --dataset first --cls 3 >0.out ',
#'python main_pretrain.py --training-fold 1 --gpu-index 5 --dataset first --cls 3 >1.out',
#'python main_pretrain.py --training-fold 6 --gpu-index 6 --dataset first --cls 3 >6.out ',
#'python main_pretrain.py --training-fold 7 --gpu-index 7 --dataset first --cls 3 >7.out',
#'python main_pretrain.py --training-fold 8 --gpu-index 2 --dataset first --cls 3 >8.out ',
#'python main_pretrain.py --training-fold 9 --gpu-index 3 --dataset first --cls 3 >9.out',



# Run multiple commands in the Conda environment
commands = [
    'python main_pretrain.py --training-fold 0 --gpu-index 4 --dataset first --cls 3 >0.out ',
    # 'python main_pretrain.py --training-fold 1 --gpu-index 5 --dataset first --cls 3 >1.out',
    # 'python main_pretrain.py --training-fold 6 --gpu-index 6 --dataset first --cls 3 >6.out ',
    # 'python main_pretrain.py --training-fold 7 --gpu-index 7 --dataset first --cls 3 >7.out',
    # 'python main_pretrain.py --training-fold 8 --gpu-index 2 --dataset first --cls 3 >8.out ',
    # 'python main_pretrain.py --training-fold 9 --gpu-index 3 --dataset first --cls 3 >9.out',
    # Add more commands as needed
]

processes = []
for command in commands:
    process = subprocess.Popen(command, shell=True)
    processes.append(process)
    print('Run the command:', command)

# Wait for all processes to finish
for process in processes:
    process.wait()

#
# # Deactivate the Conda environment
# deactivate_cmd = 'conda deactivate'
# subprocess.run(deactivate_cmd, shell=True, check=True)

# 断：18：30
