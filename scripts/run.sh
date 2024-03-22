export BASICSR_JIT=True

# For RestoreFormer
# conf_name='HQ_Dictionary'
# conf_name='RestoreFormer'

# For RestoreFormer++
#conf_name='ROHQD'
conf_name='RestoreFormerPlusPlus'

root_path='/scratch/tnguy231/RFExperiments/tnguy231'

node_n=1
ntasks_per_node=1

gpu_n=$(expr $node_n \* $ntasks_per_node)

python -u main.py \
--root-path $root_path \
--base 'configs/'$conf_name'.yaml' \
-t True \
--postfix $conf_name'_gpus'$gpu_n \
--num-nodes $node_n \
--random-seed True \
--no-test True \
--debug False \
#--enable-profiler True

#--root-path '/scratch/tnguy231/RFExperiments/tnguy231' --base 'configs/ROHQD.yaml' -t True --postfix ROHQD_gpus1 --num-nodes 1 --random-seed True --debug True
