for i in 1 2 4 8 16 32
do
srun -o ../workdirs/workdir-$i/slurm.out.txt -e ../workdirs/workdir-$i/slurm.errr.txt -n1 --gres=gpu:1 singularity exec --nv /master/home/wouyang/containers/ubuntu_tensorflow_1_5_0_pytorch_0_2_0_gpu.img python run_shareloc.py --workdir=../workdirs/workdir-$i/ &
done
