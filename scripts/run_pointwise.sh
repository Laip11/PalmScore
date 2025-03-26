CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/get_pointwise_outputs.py \
      --model_name_or_path  /nfsdata/laip/model/LLM-Research/Meta-Llama-3___1-70B-Instruct \
      --save_dir results \
      --points 5 \
      --batch_size 2 \
      --max_new_tokens 10 \
      --with_feedback 0 \
      --dtype bfloat16 \
      --max_length 2000 \
      --input_file data/point_wise/flask.json


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/get_pointwise_outputs.py \
      --model_name_or_path  /nfsdata/laip/model/LLM-Research/Meta-Llama-3___1-70B-Instruct\
      --save_dir results \
      --points 5 \
      --batch_size 2 \
      --max_new_tokens 10 \
      --with_feedback 0 \
      --dtype bfloat16 \
      --max_length 2000 \
      --input_file data/point_wise/hs_valid.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/get_pointwise_outputs.py \
      --model_name_or_path  /nfsdata/laip/model/LLM-Research/Meta-Llama-3___1-70B-Instruct\
      --save_dir results \
      --points 5 \
      --batch_size 2 \
      --max_new_tokens 10 \
      --with_feedback 0 \
      --dtype bfloat16 \
      --max_length 2000 \
      --input_file data/point_wise/helpsteer.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/get_pointwise_outputs.py \
#       --model_name_or_path  /nfsdata/laip/model/LLM-Research/Meta-Llama-3___1-70B-Instruct\
#       --save_dir results \
#       --points 5 \
#       --batch_size 2 \
#       --max_new_tokens 10 \
#       --with_feedback 0 \
#       --dtype bfloat16 \
#       --max_length 2000 \
#       --input_file data/point_wise/helpsteer.json


# model_list=(Qwen2___5-0___5B-Instruct Qwen2___5-3B-Instruct Qwen2___5-7B-Instruct Qwen2___5-14B-Instruct Qwen2___5-32B-Instruct)

# for model in ${model_list[@]}
# do
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/get_pointwise_outputs.py \
#       --model_name_or_path  /nfsdata/laip/model/Qwen/${model} \
#       --save_dir results \
#       --points 5 \
#       --batch_size 2 \
#       --max_new_tokens 10 \
#       --with_feedback 0 \
#       --dtype bfloat16 \
#       --max_length 2000 \
#       --input_file data/point_wise/helpsteer.json
# done
#CUDA_VISIBLE_DEVICES=0,1 python3 /get_pointwise_outputs.py \
#      --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
#      --save_dir results \
#      --points 5 \
#      --batch_size 8 \
#      --max_new_tokens 1024 \
#      --with_feedback 1 \
#      --dtype bfloat16 \
#      --input_file data/point_wise/flask.json

