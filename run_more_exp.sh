set -ex

## base
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_base \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn

## avg pooling
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_avg \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn \
--pooling avg

## no pooling
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_nopool \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn \
--pooling ''

## cnn_dim: 64 32 1
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_cnn64-32-1 \
--cnn_dim 64 32 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn

## cnn_dim: 64 32 1, pad 0
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_cnn64-32-1_pad0 \
--cnn_dim 64 32 1 \
--cnn_pad 0 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn \
--pooling ''

## fc_dim: 3 3 3
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_fc3-3-3 \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn

## resnet34
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_res34 \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet34 \
--pretrained_model_path pretrained_models/resnet34-333f7ec4.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn

## no cxn
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_nocxn \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5

## relu 0.2
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_relu0.2 \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn \
--relu_slope 0.2

## relu 0.8
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_relu0.8 \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn \
--relu_slope 0.8

## drop 0.5
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_drop0.5 \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.5 \
--save_epoch_freq 5 \
--use_cxn

## contrastive
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_cont0.1 \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn \
--lambda_contrastive 0.1

## contrastive
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--dataroot_val /media/ligong/Toshiba/Datasets/UTKFace \
--datafile_val data/test_pairs_m10_utk.txt \
--mode train \
--name 0801_test_regu0.1 \
--cnn_dim 64 1 \
--cnn_pad 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--dropout 0.05 \
--save_epoch_freq 5 \
--use_cxn \
--lambda_regularization 0.1
