sudo python3 main_tasn.py -a tasn -b 96 -p 50 -j 8 --lr 0.005 --dist-url 'tcp://127.0.0.1:123' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /v-helzhe/data/cub_img
