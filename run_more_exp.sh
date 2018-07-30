set -ex

## base: cnn 64 1, avg, fc 3, no cxn, reg 1, con 1
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name 729_base \
--cnn_dim 64 1 \
--fc_dim 3 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling avg \
--dropout 0.05 \
--lambda_regularization 1 \
--lambda_contrastive 1 \
--batch_size 128

## use cxn
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name 729_cxn \
--cnn_dim 64 1 \
--fc_dim 3 \
--use_cxn \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling avg \
--dropout 0.05 \
--lambda_regularization 1 \
--lambda_contrastive 1 \
--batch_size 128

## max
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name 729_base \
--cnn_dim 64 1 \
--fc_dim 3 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05 \
--lambda_regularization 1 \
--lambda_contrastive 1 \
--batch_size 128 \

## no avg, (how to extract feature at test time?)
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name 729_noavg \
--cnn_dim 64 1 \
--fc_dim 3 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling '' \
--dropout 0.05 \
--lambda_regularization 1 \
--lambda_contrastive 1 \
--batch_size 128

## cnn 256 64 1, pad 0
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name 729_cnn256-64-1_pad0 \
--cnn_dim 256 64 1 \
--cnn_pad 0 \
--fc_dim 3 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling '' \
--dropout 0.05 \
--lambda_regularization 1 \
--lambda_contrastive 1 \
--batch_size 128

## fc 3 3
python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name 729_fc3-3 \
--cnn_dim 64 1 \
--fc_dim 3 3 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling avg \
--dropout 0.05 \
--lambda_regularization 1 \
--lambda_contrastive 1 \
--batch_size 128
