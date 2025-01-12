model_name=ElxaTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 16 \
  --e_layers 2 \
  --patch_len 8 \
  --stride 4 \
  --d_model 128 \
  --d_ff 1024 \
  --learning_rate 1e-4 \
  --percent_dx 0.2 \
  --percent_dy 0.2 \
  --top_sk 40 \
  --top_dk 20 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1
