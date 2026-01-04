export CACHE_DIR="/nfshomes/asarkar6/trinity/model_weights/"
export OUTPUT_DIR="/nfshomes/asarkar6/trinity/JANe-project/embeddings/"


accelerate launch --num_processes 2 --num_machines 1 --mixed_precision fp16 --main_process_port 29501 /nfshomes/asarkar6/aditya/JANe/inference/clips/run_clip.py \
 --pretrained_model_name_or_path="openai/clip-vit-base-patch32"\
 --model_name="blip"\
 --task_type="class"\
 --output_dir=$OUTPUT_DIR\
 --cache_dir=$CACHE_DIR\
 --train_batch_size=2\
 --mixed_precision="fp16"\
 --data="coco"\