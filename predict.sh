# !/bin/bash -e
# export nnUNet_raw_data_base="/workspace/Flare23-main/dataset/nnUNet_raw_data_base"
export RESULTS_FOLDER="/workspace/Flare23-main/dataset/nnUNet_trained_models"
echo  "\e[91m start predict use FlareBaseline 2d... \e[0m"
python /workspace/Flare23-main/nnunet/inference/predict_simple.py -i '/workspace/inputs' -o '/workspace/outputs/' -t 9 -p nnUNetPlansFLARE22Small   -m 3d_fullres -tr nnUNetTrainerV2_FLARE_Small  -f all  --mode fastest --disable_tta