rem python visualize_attention.py --pretrained_weights dino_deitsmall8_pretrain_full_checkpoint.pth --image_path ../img.png --output_dir ..
rem python visualize_attention.py --image_size=224 --arch vit_small --patch_size 8 --image_path ../img.png --output_dir ..
python visualize_attention.py --arch vit_small --patch_size 8 --image_path ../img.png --output_dir ..
pause
