CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_1/strong/R1R1R4_CO_semi.yaml &
PID1=$!

CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_1/strong/R1R1R4_RG_semi.yaml &
PID2=$!

CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_1/strong/R1R1R4_B_CJ_semi.yaml &
PID3=$!

# 等待前三个进程执行完成
wait $PID1
wait $PID2
wait $PID3

# 执行后三个命令
CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_1/strong/R1R1R4_B_CO_semi.yaml &
PID4=$!

CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_1/strong/R1R1R4_B_RG_semi.yaml &
PID5=$!

CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_1/strong/R1R1R4_CJ_CO_semi.yaml &
PID6=$!

# 等待后三个进程执行完成
wait $PID4
wait $PID5
wait $PID6

CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_1/strong/R1R1R4_CJ_RG_semi.yaml &
PID7=$!
CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_1/strong/R1R1R4_CO_RG_semi.yaml &
PID8=$!
