base_dir=..
train_dir=$base_dir/AGDT/data/atsa
code_dir=thumt
work_dir=$base_dir/AGDT
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
for idx in `seq 0 1 5` 
do
    for n_t in `seq 4 1 4` 
    do
        python $work_dir/$code_dir/thumt/bin/trainer.py \
          --model rnnsearch \
          --output $work_dir/l_agdt-result-$idx \
          --input $train_dir/atsa_train_14_l.txt \
          --vocabulary $train_dir/vocab_cs_14_l.txt $train_dir/vocab_aspect_14_l.txt $train_dir/vocab_target.txt $train_dir/vocab_flag.txt $train_dir/vocab.c \
          --evaluation $train_dir/atsa_test_14_l_flag.txt \
          --parameters=device_list=[0],rnn_cell="DL4MTGRULAUTransiCell",num_transi=$n_t,use_prediction=True,use_aspect=True,use_aspect_gate=True,save_checkpoint_steps=10000,eval_steps=10,train_steps=4000,batch_size=4096,constant_batch_size=False,dropout=0.5,embedding_size=300,learning_rate=0.01,learning_rate_decay=rnnplus_warmup_decay,warmup_steps=50,s=5000,e=4000,adam_epsilon=1e-6,use_vec=True,gpu_memory_fraction=1,use_capsule_net=False,alpha=0.5,task="atsa"
    done
done
