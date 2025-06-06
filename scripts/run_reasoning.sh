CUDA_VISIBLE_DEVICES=0 python3 lager/get_pointwise_outputs.py \
      --model_name_or_path  LLM-Research/Meta-Llama-3.1-8B-Instruct \
      --save_dir results \
      --points 5 \
      --batch_size 4 \
      --max_new_tokens 256 \
      --with_feedback \
      --dtype bfloat16 \
      --input_file data/main/flask.json