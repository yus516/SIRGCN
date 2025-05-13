python src/train_sirgnn.py --horizon=1 --gpu=0 --dataset=japan --sim_mat=japan-adj --seed=420\
    > ablation/Japan-sir-train-Winter+Summer-test-Spring+Fall--ablation--420-many-beta-final.log 2>&1 &

# python src/train_sirgnn.py --horizon=1 --gpu=0 --dataset=state360 --sim_mat=state-adj-49 --seed=42\
#     > ablation/US-sir-train-Winter+Summer-test-Spring+Fall--ablation--42-beta-divided-10-single-final.log 2>&1 &