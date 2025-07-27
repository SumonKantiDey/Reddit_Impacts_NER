for seed in 3 4 5 6
do
python -m src.train_plm_crf_new \
    --gpus "7" \
    --model_name "microsoft/deberta-large" \
    --learning_rate 2e-5 \
    --dropout 0.2 \
    --batch_size 16 \
    --num_epochs 10 \
    --weight_decay 0.01 \
    --logging_steps 20 \
    --eval_steps 20 \
    --save_steps 100 \
    --lr_scheduler_type "linear" \
    --max_grad_norm 1.0 \
    --early_stopping_patience 3 \
    --use_crf True \
    --seed $seed


python -m src.test_plm_crf \
    --gpus "7" \
    --model_name "microsoft/deberta-large" \
    --batch_size 8 \
    --num_epochs 5 \
    --weight_decay 0.01 \
    --logging_steps 20 \
    --eval_steps 20 \
    --use_crf True \
    --seed $seed

python -m src.confidence_interval \
    --model_name "microsoft/deberta-large" \
    --use_crf True \
    --seed $seed
done
