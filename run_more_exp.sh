set -ex

# python main.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train_pairs_utk10.txt \
# --mode train \
# --name new_siamese_utk_avg_641_2_r100 \
# --cnn_dim 64 1 \
# --fc_dim 2 \
# --which_model resnet18 \
# --use_pretrained_model \
# --pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
# --age_bins 1 11 21 31 41 51 61 71 81 91 \
# --num_classes 3 \
# --lr 0.0002 \
# --init_type kaiming \
# --pooling avg \
# --dropout 0.05 \
# --lambda_reg 100

# python main.py \
# --dataroot /media/ligong/Toshiba/Datasets/UTKFace \
# --datafile data/train_pairs_utk10.txt \
# --mode train \
# --name new_siamese_utk_avg_641_2 \
# --cnn_dim 64 1 \
# --fc_dim 2 \
# --which_model resnet18 \
# --use_pretrained_model \
# --pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
# --age_bins 1 11 21 31 41 51 61 71 81 91 \
# --num_classes 3 \
# --lr 0.0002 \
# --init_type kaiming \
# --pooling avg \
# --dropout 0.05 \
# --lambda_reg 0

python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name new_siamese_m10_utk_643_3 \
--cnn_dim 64 3 \
--fc_dim 3 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 11 21 31 41 51 61 71 81 91 \
--num_classes 3 \
--lr 0.0002 \
--init_type kaiming \
--pooling max \
--dropout 0.05

python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name new_siamese_m10_utk_643_3_r1 \
--cnn_dim 64 3 \
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
--lambda_reg 1

python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name new_siamese_m10_utk_643_3_r10 \
--cnn_dim 64 3 \
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
--lambda_reg 10

python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name new_siamese_m10_utk_643_3_r100 \
--cnn_dim 64 3 \
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
--lambda_reg 100


python main.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train_pairs_m10_utk.txt \
--mode train \
--name new_siamese_m10_utk_641_3 \
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
--dropout 0.05
