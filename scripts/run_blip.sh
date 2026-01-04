export CACHE_DIR="/nfshomes/asarkar6/trinity/model_weights/"
export OUTPUT_DIR="/nfshomes/asarkar6/trinity/JANe-project/embeddings/"


accelerate launch --num_processes 2 --num_machines 1 --mixed_precision fp16 --main_process_port 29501 /nfshomes/asarkar6/aditya/JANe/inference/lvlms/run_blip.py \
 --pretrained_model_name_or_path="Salesforce/blip-image-captioning-large"\
 --model_name="blip"\
 --output_dir=$OUTPUT_DIR\
 --cache_dir=$CACHE_DIR\
 --train_batch_size=2\
 --mixed_precision="fp16"\
 --data="coco"\