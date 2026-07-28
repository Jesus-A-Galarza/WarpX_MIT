# WarpX for SubMIT
## Instalation
### GPU
First we clone the repository

```bash
git clone https://github.com/BLAST-WarpX/warpx.git warpx
cd warpx
```
Then, the compilation must be made in a GPU node:

```bash
srun --partition=submit-gpu --gres=gpu:1 --cpus-per-gpu=8 --mem=64G --time=02:00:00 --pty bash
```
Then we set up the environment variables:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/releases/LCG_108_cuda/hdf5_mpi/1.14.6/x86_64-el9-gcc13-opt/hdf5_mpi-env.sh
export CUDA_ROOT=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/lib:/usr/local/cuda/lib
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
export CC=$(which gcc)
export CXX=$(which g++)
export FC=$(which gfortran)
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=$(which g++)
```
