seed=1
base_n_embd=192
# for n_embd in 48 96 192 384
# for n_embd in 192 384 768
for n_embd in 48 96
do
for lr in 3.2e-4 6.4e-4 1.28e-3 2.56e-3 5.12e-3 1.024e-2 2.048e-2 4.096e-2 8.192e-2 1.6384e-1 3.2768e-1 6.5536e-1 1.31072e0
# for lr in 6.5536e-1 1.31072e0
# for lr in 2e-5 4e-5 8e-5 1.6e-4 3.2e-4 6.4e-4 1.28e-3 2.56e-3 5.12e-3 1.024e-2 
do
ACCUM_STEPS=32
# max_iters=18000
max_iters=5000
lbatch -g 1 -t 24 \
    -q gpu-preempt --name owt_wd_$lr \
    --memory 100 -c 8 --gputype "[l40s|a100]" --cmd \
    python train.py config/train_gpt2_small_block256.py --compile=False \
    --n_embd=$n_embd --base_shape_file=gpt_${base_n_embd}.bsh --base_n_embd=$base_n_embd \
    --batch_size=8 --gradient_accumulation_steps=$ACCUM_STEPS --learning_rate=$lr \
    --max_iters=$max_iters --weight_decay=0.01 \
    --wandb_run_name=mup_owt_gpt2small_block256_$lr_ --random_seed=$seed --use_cosine_decay=True &
    # python main.py --weight_decay $wd --seed $seed
done
done