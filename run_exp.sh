set -e

# ## utk5, simple
# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_simple364 \
# --pretrained_model_path pretrained_models/alexnet.pth \
# --which_model n_layers \
# --n_layers 3 \
# --nf 64 \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002 \
# --loadSize 128

# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_simple332 \
# --pretrained_model_path pretrained_models/alexnet.pth \
# --which_model n_layers \
# --n_layers 3 \
# --nf 32 \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002 \
# --loadSize 128

# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_simple316 \
# --pretrained_model_path pretrained_models/alexnet.pth \
# --which_model n_layers \
# --n_layers 3 \
# --nf 16 \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002 \
# --loadSize 128

# ## alexlite fix
# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_alexlite_fix \
# --use_pretrained_model \
# --pretrained_model_path pretrained_models/alexnet.pth \
# --pooling avg \
# --which_model alexnet_lite \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002 \
# --finetune_fc_only

# ## alexlite max
# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_alexlite_max \
# --use_pretrained_model \
# --pretrained_model_path pretrained_models/alexnet.pth \
# --pooling max \
# --which_model alexnet_lite \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002

# ## utk5, resnet18
# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_resnet18 \
# --which_model resnet18 \
# --use_pretrained_model \
# --pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002 \
# --init_type kaiming

# cd ../face-aging
# pwd

# # faceaing, base for UTK
# python train.py \
# --model faceaging \
# --dataset_mode faceaging \
# --age_binranges 1 21 41 61 81 \
# --embedding_nc 5 \
# --train_label_pairs data/train_label_pair.txt \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --sourcefile_A ./data/UTK_train.txt \
# --name utk5_base \
# --which_model_netG resnet_6blocks \
# --which_model_netD n_layers \
# --n_layers_D 4 \
# --which_model_netIP alexnet \
# --lr 0.0002 \
# --lambda_L1 0.0 \
# --lambda_IP 0.8 \
# --pretrained_model_path_IP pretrained_models/alexnet.pth \
# --pretrained_model_path_AC pretrained_models/AC_alexnet_lite_avg_utk5.pth \
# --lr_AC 0.0002 \
# --pool_size 0 \
# --loadSize 128 \
# --fineSize 128 \
# --fineSize_IP 224 \
# --fineSize_AC 224 \
# --display_port 8097 \
# --display_freq 10 \
# --lambda_AC 0.8 \
# --batchSize 6 \
# --max_dataset_size 4096 \
# --display_aging_visuals \
# --pooling_AC avg \
# --no_AC_on_fake \
# --niter_decay 50 \
# --verbose


# # n_layers_D 3
# python train.py \
# --model faceaging \
# --dataset_mode faceaging \
# --age_binranges 11 21 31 41 51 \
# --embedding_nc 5 \
# --train_label_pairs data/train_label_pair.txt \
# --dataroot /media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400_splitted/ \
# --name cacd5_D3 \
# --which_model_netG resnet_6blocks \
# --which_model_netD n_layers \
# --n_layers_D 3 \
# --which_model_netIP alexnet \
# --lr 0.0002 \
# --lambda_L1 0.0 \
# --lambda_IP 0.8 \
# --pretrained_model_path_IP pretrained_models/alexnet.pth \
# --pretrained_model_path_AC pretrained_models/AC_alexnet_lite_avg_cacd5.pth \
# --lr_AC 0.0002 \
# --pool_size 0 \
# --loadSize 128 \
# --fineSize 128 \
# --fineSize_IP 224 \
# --fineSize_AC 224 \
# --display_port 8097 \
# --display_freq 10 \
# --lambda_AC 0.8 \
# --batchSize 6 \
# --max_dataset_size 4096 \
# --display_aging_visuals \
# --pooling_AC avg \
# --no_AC_on_fake \
# --niter_decay 50 \
# --verbose

# # no_dropout
# python train.py \
# --model faceaging \
# --dataset_mode faceaging \
# --age_binranges 11 21 31 41 51 \
# --embedding_nc 5 \
# --train_label_pairs data/train_label_pair.txt \
# --dataroot /media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400_splitted/ \
# --name cacd5_nodrop \
# --which_model_netG resnet_6blocks \
# --which_model_netD n_layers \
# --n_layers_D 4 \
# --which_model_netIP alexnet \
# --lr 0.0002 \
# --lambda_L1 0.0 \
# --lambda_IP 0.8 \
# --pretrained_model_path_IP pretrained_models/alexnet.pth \
# --pretrained_model_path_AC pretrained_models/AC_alexnet_lite_avg_cacd5.pth \
# --lr_AC 0.0002 \
# --pool_size 0 \
# --loadSize 128 \
# --fineSize 128 \
# --fineSize_IP 224 \
# --fineSize_AC 224 \
# --display_port 8097 \
# --display_freq 10 \
# --lambda_AC 0.8 \
# --batchSize 6 \
# --max_dataset_size 4096 \
# --display_aging_visuals \
# --pooling_AC avg \
# --no_AC_on_fake \
# --niter_decay 50 \
# --no_dropout \
# --verbose

# # lambda_AC 8
# python train.py \
# --model faceaging \
# --dataset_mode faceaging \
# --age_binranges 11 21 31 41 51 \
# --embedding_nc 5 \
# --train_label_pairs data/train_label_pair.txt \
# --dataroot /media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400_splitted/ \
# --name cacd5_AC8 \
# --which_model_netG resnet_6blocks \
# --which_model_netD n_layers \
# --n_layers_D 4 \
# --which_model_netIP alexnet \
# --lr 0.0002 \
# --lambda_L1 0.0 \
# --lambda_IP 0.8 \
# --pretrained_model_path_IP pretrained_models/alexnet.pth \
# --pretrained_model_path_AC pretrained_models/AC_alexnet_lite_avg_cacd5.pth \
# --lr_AC 0.0002 \
# --pool_size 0 \
# --loadSize 128 \
# --fineSize 128 \
# --fineSize_IP 224 \
# --fineSize_AC 224 \
# --display_port 8097 \
# --display_freq 10 \
# --lambda_AC 8 \
# --batchSize 6 \
# --max_dataset_size 4096 \
# --display_aging_visuals \
# --pooling_AC avg \
# --no_AC_on_fake \
# --niter_decay 50 \
# --verbose

# # ACfake
# python train.py \
# --model faceaging \
# --dataset_mode faceaging \
# --age_binranges 11 21 31 41 51 \
# --embedding_nc 5 \
# --train_label_pairs data/train_label_pair.txt \
# --dataroot /media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400_splitted/ \
# --name cacd5_ACfake \
# --which_model_netG resnet_6blocks \
# --which_model_netD n_layers \
# --n_layers_D 4 \
# --which_model_netIP alexnet \
# --lr 0.0002 \
# --lambda_L1 0.0 \
# --lambda_IP 0.8 \
# --pretrained_model_path_IP pretrained_models/alexnet.pth \
# --pretrained_model_path_AC pretrained_models/AC_alexnet_lite_avg_cacd5.pth \
# --lr_AC 0.0002 \
# --pool_size 0 \
# --loadSize 128 \
# --fineSize 128 \
# --fineSize_IP 224 \
# --fineSize_AC 224 \
# --display_port 8097 \
# --display_freq 10 \
# --lambda_AC 0.8 \
# --batchSize 6 \
# --max_dataset_size 4096 \
# --display_aging_visuals \
# --pooling_AC avg \
# --niter_decay 50 \
# --verbose

# # lambda_IP 0.01
# python train.py \
# --model faceaging \
# --dataset_mode faceaging \
# --age_binranges 11 21 31 41 51 \
# --embedding_nc 5 \
# --train_label_pairs data/train_label_pair.txt \
# --dataroot /media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400_splitted/ \
# --name cacd5_IP001 \
# --which_model_netG resnet_6blocks \
# --which_model_netD n_layers \
# --n_layers_D 4 \
# --which_model_netIP alexnet \
# --lr 0.0002 \
# --lambda_L1 0.0 \
# --lambda_IP 0.01 \
# --pretrained_model_path_IP pretrained_models/alexnet.pth \
# --pretrained_model_path_AC pretrained_models/AC_alexnet_lite_avg_cacd5.pth \
# --lr_AC 0.0002 \
# --pool_size 0 \
# --loadSize 128 \
# --fineSize 128 \
# --fineSize_IP 224 \
# --fineSize_AC 224 \
# --display_port 8097 \
# --display_freq 10 \
# --lambda_AC 0.8 \
# --batchSize 6 \
# --max_dataset_size 4096 \
# --display_aging_visuals \
# --pooling_AC avg \
# --no_AC_on_fake \
# --niter_decay 50 \
# --verbose


# cd ../face-siamese
# pwd

# ## utk5, resnet34
# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_resnet34 \
# --which_model resnet34 \
# --use_pretrained_model \
# --pretrained_model_path pretrained_models/resnet34-333f7ec4.pth \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002 \
# --init_type kaiming

# ## utk5, resnet50
# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_resnet50 \
# --which_model resnet50 \
# --use_pretrained_model \
# --pretrained_model_path pretrained_models/resnet50-19c8e357.pth \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002 \
# --init_type kaiming

# ## utk5, resnet101
# python main_regression.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train.txt \
# --mode train \
# --name utk5_resnet101 \
# --which_model resnet101 \
# --use_pretrained_model \
# --pretrained_model_path pretrained_models/resnet101-5d3b4d8f.pth \
# --age_bins 1 21 41 61 81 \
# --num_classes 5 \
# --weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
# --lr 0.0002 \
# --init_type kaiming

python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_644_4 \
--cnn_dim 64 4 \
--fc_dim 4 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05

# 72
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_641_2 \
--cnn_dim 64 1 \
--fc_dim 2 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05

# 74.85
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_648_8 \
--cnn_dim 64 8 \
--fc_dim 8 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05

# 70.40
# 7.002   5.127
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_641_2_r1 \
--cnn_dim 64 1 \
--fc_dim 2 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05 \
--lambda_reg 1

# 70.25
# 0.7381   0.8940
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_641_2_r01 \
--cnn_dim 64 1 \
--fc_dim 2 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05 \
--lambda_reg 0.1

# 68.90
# 3.3863   2.7664
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_641_2_r001 \
--cnn_dim 64 1 \
--fc_dim 2 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05 \
--lambda_reg 0.01

# 67.95
# 7.2494   7.9706
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_641_2_r0001 \
--cnn_dim 64 1 \
--fc_dim 2 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05 \
--lambda_reg 0.001

# 72.05
# 3.768   3.015
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_641_2_r10 \
--cnn_dim 64 1 \
--fc_dim 2 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05 \
--lambda_reg 10


python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_utk10.txt \
--mode train \
--name new_siamese_utk_641_2_r100 \
--cnn_dim 64 1 \
--fc_dim 2 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05 \
--lambda_reg 100