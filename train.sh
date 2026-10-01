CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=16085 \
  --nproc_per_node=4 \
  train.py \
  -c 'configs/train_convnext_s.yml' \
  --use-amp \
  --seed=0