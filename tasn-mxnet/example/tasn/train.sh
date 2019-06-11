export MXNET_CPU_WORKER_NTHREADS=96
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./nccl/build/lib/:./cuda/lib64/
python train.py --gpus 0,1,2,3,4,5,6,7 \
	--model-prefix ./model/tasn \
	--data-nthreads 128 \
    --batch-size 96 --num-classes 200 --num-examples 5994
