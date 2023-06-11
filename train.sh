export od_semi_setting='refuge_od/1_4/split_0'
export oc_semi_setting='refuge_oc/1_4/split_0'
export od_oc_semi_setting='refuge_od_oc/1_4/split_0'

CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
  --dataset refuge_od --data-root ./data/fundus_datasets/od_oc/REFUGE/ \
  --batch-size 8 --backbone resnet50 --model deeplabv3plus \
  --labeled-id-path dataset/splits/$od_semi_setting/labeled.txt \
  --unlabeled-id-path dataset/splits/$od_semi_setting/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks/$od_semi_setting \
  --save-path outdir/models/$od_semi_setting \
  --lr 0.002 & \
CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
  --dataset refuge_oc --data-root ./data/fundus_datasets/od_oc/REFUGE/ \
  --batch-size 8 --backbone resnet50 --model deeplabv3plus \
  --labeled-id-path dataset/splits/$oc_semi_setting/labeled.txt \
  --unlabeled-id-path dataset/splits/$oc_semi_setting/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks/$oc_semi_setting \
  --save-path outdir/models/$oc_semi_setting \
  --lr 0.002 & \
CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
  --dataset refuge_od_oc --data-root ./data/fundus_datasets/od_oc/REFUGE/ \
  --batch-size 8 --backbone resnet50 --model deeplabv3plus \
  --labeled-id-path dataset/splits/$od_oc_semi_setting/labeled.txt \
  --unlabeled-id-path dataset/splits/$od_oc_semi_setting/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks/$od_oc_semi_setting \
  --save-path outdir/models/$od_oc_semi_setting \
  --lr 0.002 & 