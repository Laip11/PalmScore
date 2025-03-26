# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/get_pointwise_outputs.py \
#       --model_name_or_path  /nfsdata/laip/model/Qwen/Qwen2___5-72B-Instruct \
#       --save_dir results \
#       --points 5 \
#       --batch_size 2 \
#       --max_new_tokens 10 \
#       --with_feedback 0 \
#       --dtype bfloat16 \
#       --max_length 2000 \
#       --input_file data/point_wise/flask.json


# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/get_pointwise_outputs.py \
#       --model_name_or_path  /nfsdata/laip/model/Qwen/Qwen2___5-72B-Instruct\
#       --save_dir results \
#       --points 5 \
#       --batch_size 2 \
#       --max_new_tokens 10 \
#       --with_feedback 0 \
#       --dtype bfloat16 \
#       --max_length 2000 \
#       --input_file data/point_wise/hs_valid.jsonl

# model_list=(/nfsdata/laip/model/Qwen/Qwen2___5-32B-Instruct \
#             /nfsdata/laip/model/Qwen/Qwen2___5-0___5B-Instruct \
#             /nfsdata/laip/model/Qwen/Qwen2___5-3B-Instruct \
#             /nfsdata/laip/model/Qwen/Qwen2___5-7B-Instruct \
#             /nfsdata/laip/model/Qwen/Qwen2___5-14B-Instruct )
model_list=(/nfsdata/laip/model/LLM-Research/Meta-Llama-3___1-70B-Instruct)

for model in ${model_list[@]}
do

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 src/get_pointwise_outputs.py \
      --model_name_or_path  ${model} \
      --save_dir results \
     --points 5 \
     --batch_size 2 \
     --max_new_tokens 1024 \
     --with_feedback 1 \
     --dtype bfloat16 \
     --input_file data/point_wise/hs_valid.jsonl


CUDA_VISIBLE_DEVICES=4,5,6,7 python3 src/get_pointwise_outputs.py \
      --model_name_or_path  ${model} \
      --save_dir results \
     --points 5 \
     --batch_size 2 \
     --max_new_tokens 1024 \
     --with_feedback 1 \
     --dtype bfloat16 \
     --input_file data/point_wise/flask.json

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 src/get_pointwise_outputs.py \
      --model_name_or_path  ${model} \
      --save_dir results \
     --points 5 \
     --batch_size 2 \
     --max_new_tokens 1024 \
     --with_feedback 1 \
     --dtype bfloat16 \
     --input_file data/point_wise/helpsteer.json

done

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 /get_pointwise_outputs.py \
#      --model_name_or_path /nfsdata/laip/model/Qwen/Qwen2___5-72B-Instruct \
#      --save_dir results \
#      --points 5 \
#      --batch_size 1 \
#      --max_new_tokens 1024 \
#      --with_feedback 1 \
#      --dtype bfloat16 \
#      --input_file data/point_wise/flask.json

