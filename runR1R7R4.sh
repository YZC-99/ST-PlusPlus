CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_7/strong/R1R7R4_B_RG_semi.yaml &
PID1=$!

CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/semi/1_7/strong/R1R7R4_CJ_CO_semi.yaml &
PID2=$!

CUDA_VISIBLE_DEVICES=1 python -W ignore main_cfg.py --config-file configs/semi/1_7/strong/R1R7R4_CJ_RG_semi.yaml &
PID3=$!

CUDA_VISIBLE_DEVICES=1 python -W ignore main_cfg.py --config-file configs/semi/1_7/strong/R1R7R4_CO_RG_semi.yaml &
PID4=$!

CUDA_VISIBLE_DEVICES=2 python -W ignore main_cfg.py --config-file configs/semi/1_7/strong/R1R7R4_B_CJ_CO_semi.yaml &
PID5=$!

CUDA_VISIBLE_DEVICES=2 python -W ignore main_cfg.py --config-file configs/semi/1_7/strong/R1R7R4_B_CJ_RG_semi.yaml &
PID6=$!

CUDA_VISIBLE_DEVICES=3 python -W ignore main_cfg.py --config-file configs/semi/1_7/strong/R1R7R4_B_CO_RG_semi.yaml &
PID7=$!

CUDA_VISIBLE_DEVICES=3 python -W ignore main_cfg.py --config-file configs/semi/1_7/strong/R1R7R4_CJ_CO_RG_semi.yaml &
PID8=$!

# 等待前三个进程执行完成
wait $PID1
wait $PID2
wait $PID3
wait $PID4
wait $PID5
wait $PID6
wait $PID7
wait $PID8

#shutdown now


