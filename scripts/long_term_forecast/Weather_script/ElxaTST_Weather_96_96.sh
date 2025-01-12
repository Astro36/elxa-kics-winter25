model_name=ElxaTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --e_layers 3 \
  --patch_len 16 \
  --stride 8 \
  --d_model 512 \
  --d_ff 1024 \
  --learning_rate 1e-4 \
  --percent_dx 0.2 \
  --percent_dy 0.4 \
  --top_sk 40 \
  --top_dk 40 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1
