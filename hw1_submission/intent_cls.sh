# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 ./test_intent.py --test_file "${1}" --ckpt_path ckpt/intent/bestPerformanceModel.pth --pred_file "${2}" --hidden_size 512 --rnn_method GRU --max_len 35 --drop_rate 0.4