set -e

## utk5, simple
python main_regression.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train.txt \
--mode train \
--name utk5_simple364 \
--pretrained_model_path pretrained_models/alexnet.pth \
--which_model n_layers \
--n_layers 3 \
--nf 64 \
--age_bins 1 21 41 61 81 \
--num_classes 5 \
--weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
--lr 0.0002 \
--loadSize 128

python main_regression.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train.txt \
--mode train \
--name utk5_simple332 \
--pretrained_model_path pretrained_models/alexnet.pth \
--which_model n_layers \
--n_layers 3 \
--nf 32 \
--age_bins 1 21 41 61 81 \
--num_classes 5 \
--weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
--lr 0.0002 \
--loadSize 128

python main_regression.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train.txt \
--mode train \
--name utk5_simple316 \
--pretrained_model_path pretrained_models/alexnet.pth \
--which_model n_layers \
--n_layers 3 \
--nf 16 \
--age_bins 1 21 41 61 81 \
--num_classes 5 \
--weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
--lr 0.0002 \
--loadSize 128

## alexlite fix
python main_regression.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train.txt \
--mode train \
--name utk5_alexlite_fix \
--use_pretrained_model \
--pretrained_model_path pretrained_models/alexnet.pth \
--pooling avg \
--which_model alexnet_lite \
--age_bins 1 21 41 61 81 \
--num_classes 5 \
--weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
--lr 0.0002 \
--finetune_fc_only

## alexlite max
python main_regression.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train.txt \
--mode train \
--name utk5_alexlite_max \
--use_pretrained_model \
--pretrained_model_path pretrained_models/alexnet.pth \
--pooling max \
--which_model alexnet_lite \
--age_bins 1 21 41 61 81 \
--num_classes 5 \
--weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
--lr 0.0002

## utk5, resnet18
python main_regression.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train.txt \
--mode train \
--name utk5_resnet18 \
--which_model resnet18 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 21 41 61 81 \
--num_classes 5 \
--weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
--lr 0.0002 \
--init_type kaiming

## utk5, resnet34
python main_regression.py \
--dataroot /media/ligong/Toshiba/Datasets/UTKFace \
--datafile data/train.txt \
--mode train \
--name utk5_resnet34 \
--which_model resnet34 \
--use_pretrained_model \
--pretrained_model_path pretrained_models/resnet18-5c106cde.pth \
--age_bins 1 21 41 61 81 \
--num_classes 5 \
--weight 0.07051924390273953 0.028346141675991896 0.0797124276358269 0.18505130615942367 0.6363708806260181 \
--lr 0.0002 \
--init_type kaiming


cd ../face-aging
pwd

# unet_128:
python train.py \
--model faceaging \
--dataset_mode faceaging \
--age_binranges 11 21 31 41 51 \
--embedding_nc 5 \
--train_label_pairs data/train_label_pair.txt \
--dataroot /media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400_splitted/ \
--name cacd5_unet128 \
--which_model_netG unet_128 \
--which_model_netD n_layers \
--n_layers_D 4 \
--which_model_netIP alexnet \
--lr 0.0002 \
--lambda_L1 0.0 \
--lambda_IP 0.8 \
--IP_pretrained_model_path pretrained_models/alexnet.pth \
--AC_pretrained_model_path pretrained_models/AC_alexnet_lite_avg_cacd5.pth \
--lr_AC 0.0002 \
--pool_size 0 \
--loadSize 128 \
--fineSize 128 \
--fineSize_IP 224 \
--fineSize_AC 224 \
--display_port 8097 \
--display_freq 10 \
--lambda_AC 0.8 \
--batchSize 6 \
--max_dataset_size 4096 \
--display_aging_visuals \
--pooling_AC avg \
--no_AC_on_fake \
--niter_decay 50 \
--verbose

# pool
python train.py \
--model faceaging \
--dataset_mode faceaging \
--age_binranges 11 21 31 41 51 \
--embedding_nc 5 \
--train_label_pairs data/train_label_pair.txt \
--dataroot /media/ligong/Toshiba/Datasets/CACD/CACD_cropped_400_splitted/ \
--name cacd5_pool \
--which_model_netG resnet_6blocks \
--which_model_netD n_layers \
--n_layers_D 4 \
--which_model_netIP alexnet \
--lr 0.0002 \
--lambda_L1 0.0 \
--lambda_IP 0.8 \
--IP_pretrained_model_path pretrained_models/alexnet.pth \
--AC_pretrained_model_path pretrained_models/AC_alexnet_lite_avg_cacd5.pth \
--lr_AC 0.0002 \
--pool_size 50 \
--loadSize 128 \
--fineSize 128 \
--fineSize_IP 224 \
--fineSize_AC 224 \
--display_port 8097 \
--display_freq 10 \
--lambda_AC 0.8 \
--batchSize 6 \
--max_dataset_size 4096 \
--display_aging_visuals \
--pooling_AC avg \
--no_AC_on_fake \
--niter_decay 50 \
--verbose
