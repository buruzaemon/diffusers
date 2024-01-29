export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./photos_16"
export OUTPUT_DIR="./stable-diffusion-v1-5-dreambooth-photos-16-epochs-80"

accelerate launch train_dreambooth.py   --gradient_checkpointing   --use_8bit_adam   --pretrained_model_name_or_path=$MODEL_NAME    --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR   --instance_prompt="a photo of sks man"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_train_epochs=80   --disable_flash_sdp   --push_to_hub
