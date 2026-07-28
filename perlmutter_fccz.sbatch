#!/bin/bash -l

# Copyright 2021-2022 Axel Huebl, Kevin Gott
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL


#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH -J WarpX
#SBATCH -A m4272_g
#SBATCH -q premium
#S BATCH -C gpu
#SBATCH -C gpu&hbm80g
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# executable & inputs fil_e or python interpreter & PICMI script here
EXE=$HOME/src/warpx/build_gg/bin/warpx.3d
INPUTS=inputs.txt

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export OMP_NUM_THREADS=16

# GPU-aware MPI optimizations
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun --cpu-bind=cores bash -c "
    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    ${EXE} ${INPUTS} ${GPU_AWARE_MPI} warpx.numprocs= 1 1 8" \
  > output.txt
