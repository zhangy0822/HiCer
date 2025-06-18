DATASET_NAME='coco'
DATA_PATH='/home/zy/dev/datasets/'${DATASET_NAME}
TRAIN_MODE='r2g'

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} \
  --num_epochs=15 --lr_update=10 --learning_rate=.0001 --precomp_enc_type basic --workers 22 \
  --batch_size 128 \
  --log_step 200 --val_step 1000 --embed_size 1024 \
  --use_moco 1 --moco_M 4096 --moco_r 0.999 --loss_lamda 1 \
  --mu 90 --gama 0.5 \
  --n_layer 3 --d_k 128 --d_v 128 --head 8 --train_mode ${TRAIN_MODE} \
  --gpo_step 40 \
  --use_angle_loss 0 \
  --angle_loss_ratio 0.01
