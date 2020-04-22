for seed in $(seq 0 4);do
    CUDA_VISIBLE_DEVICES=2 python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e7 --seed=${seed}  --num_env=10 --log_path=/data4/shuwd/logs/pong-ppo2-mirrorloss-notuse/seed-${seed};
done
