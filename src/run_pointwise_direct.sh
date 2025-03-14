CUDA_VISIBLE_DEVICES=0 python3 get_pointwise_outputs.py \
      --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
      --save_dir results \
      --points 5 \
      --batch_size 4 \
      --max_new_tokens 10 \
      --with_feedback 0 \
      --dtype bfloat16 \
      --max_length 2000 \
      --input_file data/flask.json
