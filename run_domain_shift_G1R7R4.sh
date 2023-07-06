# whole
CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_B_CJ_CO_RG_semi.yaml &
PID1=$!
#None
CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_None_semi.yaml &
PID2=$!

#default
CUDA_VISIBLE_DEVICES=1 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/R1R7R4_semi.yaml &
PID3=$!

#One
CUDA_VISIBLE_DEVICES=1 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_B_semi.yaml &
PID4=$!
CUDA_VISIBLE_DEVICES=2 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_CJ_semi.yaml &
PID5=$!
CUDA_VISIBLE_DEVICES=2 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_CO_semi.yaml &
PID6=$!
CUDA_VISIBLE_DEVICES=3 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_RG_semi.yaml &
PID7=$!

#Two
CUDA_VISIBLE_DEVICES=3 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_B_CJ_semi.yaml &
PID8=$!

# waiting
wait $PID1
wait $PID2
wait $PID3
wait $PID4
wait $PID5
wait $PID6
wait $PID7
wait $PID8

CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_B_CO_semi.yaml &
PID9=$!
CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_B_RG_semi.yaml &
PID10=$!
CUDA_VISIBLE_DEVICES=1 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_CJ_CO_semi.yaml &
PID11=$!
CUDA_VISIBLE_DEVICES=1 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_CJ_RG_semi.yaml &
PID12=$!
CUDA_VISIBLE_DEVICES=2 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_CO_RG_semi.yaml &
PID13=$!

#Three
CUDA_VISIBLE_DEVICES=2 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_B_CJ_CO_semi.yaml &
PID14=$!
CUDA_VISIBLE_DEVICES=3 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_B_CJ_RG_semi.yaml &
PID15=$!
CUDA_VISIBLE_DEVICES=3 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_B_CO_RG_semi.yaml &
PID16=$!

wait PID9
CUDA_VISIBLE_DEVICES=0 python -W ignore main_cfg.py --config-file configs/domain_shift_semi/1_7/strong/G1R7R4_CJ_CO_RG_semi.yaml &
PID17=$!


# 等待前三个进程执行完成
wait $PID9
wait $PID10
wait $PID11
wait $PID12
wait $PID13
wait $PID14
wait $PID15
wait $PID16
wait $PID17

#shutdown now


