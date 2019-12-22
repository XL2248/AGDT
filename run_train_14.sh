base_dir=..
train_dir=$base_dir/AGDT/data/acsa
code_dir=thumt
work_dir=$base_dir/AGDT
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6
for idx in `seq 0 1 5` 
do
    for n_t in `seq 4 1 4` 
    do
	    python $work_dir/$code_dir/thumt/bin/trainer.py \
	       --model rnnsearch \
	       --output $work_dir/14_agdt-result-$idx \
	       --input $train_dir/acsa_train_14.txt \
	       --vocabulary $train_dir/vocab_cs_14.txt $train_dir/vocab_aspect_14.txt $train_dir/vocab_target_14.txt $train_dir/vocab_flag.txt $train_dir/vocab.c \
	       --evaluation $train_dir/acsa_test_14_flag.txt \
	       --parameters=device_list=[0],rnn_cell="DL4MTGRULAUTransiCell",num_transi=$n_t,use_char_feature=False,use_vec_a=True,char_embedding_size=128,dropout=0.5,use_prediction=True,use_aspect=True,use_aspect_gate=True,save_checkpoint_steps=10000,eval_steps=10,train_steps=4500,batch_size=4096,max_length=128,constant_batch_size=False,embedding_size=300,learning_rate=1e-2,learning_rate_decay=rnnplus_warmup_decay,warmup_steps=50,s=5000,e=4000,adam_epsilon=1e-6,use_vec=True,gpu_memory_fraction=0.12,use_capsule_net=False,alpha=0.4,task="acsa"
	done
done
