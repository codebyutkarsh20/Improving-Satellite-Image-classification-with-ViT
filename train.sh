# python main.py \
#  --exp_name=trial \
#  --add_cross_entropy \
#  --add_attn_loss \
#  --use_triplet_loss \
#  --add_triplet_loss \
#  --model_type base_imagenet

# Baseline

python main.py --exp_name=baseline_without_attn_loss --add_cross_entropy --model_type base_imagenet

python main.py --exp_name=baseline_triplet_normal --add_cross_entropy --use_triplet_loss --add_triplet_loss --model_type base_imagenet

# LORA
python main.py --exp_name=lora_without_attn_loss --add_cross_entropy --model_type lora

python main.py --exp_name=lora_with_attn_loss --add_cross_entropy --add_attn_loss --model_type lora

python main.py --exp_name=lora_all_three --add_cross_entropy --add_attn_loss --use_triplet_loss --add_triplet_loss --model_type lora

python main.py --exp_name=lora_triplet_normal --add_cross_entropy  --use_triplet_loss --add_triplet_loss --model_type lora

# LORA_MOD

python main.py --exp_name=lora_mod_without_attn_loss --add_cross_entropy  --model_type lora_mod

python main.py --exp_name=lora_mod_with_attn_loss --add_cross_entropy --add_attn_loss --model_type lora_mod

python main.py --exp_name=lora_mod_all_three --add_cross_entropy --add_attn_loss --use_triplet_loss --add_triplet_loss --model_type lora_mod

python main.py --exp_name=lora_mod_triplet_normal --add_cross_entropy --use_triplet_loss --add_triplet_loss --model_type lora_mod





