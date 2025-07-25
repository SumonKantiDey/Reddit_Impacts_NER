

# --gpus "7" this is the GPU number 

for seed in 3 4 5 6
do
python -m src.train_plm \
    --gpus "7" \
    --model_name "debarta-large" \
    --learning_rate 3e-5 \
    --dropout 0.1 \
    --batch_size 16 \
    --num_epochs 5 \
    --weight_decay 0.01 \
    --logging_steps 20 \
    --eval_steps 20 \
    --save_steps 100 \
    --lr_scheduler_type "linear" \
    --max_grad_norm 1.0 \
    --early_stopping_patience 3 \
    --seed $seed

python -m src.test_data_eval \
    --model_name "debarta-large" \
    --gpus "7" \
    --seed $seed

python -m src.confidence_interval \
    --model_name "debarta-large" \
    --seed $seed
done

