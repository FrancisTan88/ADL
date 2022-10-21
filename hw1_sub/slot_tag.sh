# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --data_dir "${1}" --ckpt_dir ckpt/slot/bestPerformanceModel.pth --pred_file "${2}" --hidden_size 512 --rnn_method LSTM --max_len 40 --drop_rate 0.4 --batch_size 64 