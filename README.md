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
We proceed by creating the build directory

```bash
cmake -S . -B build_py \
  -DCMAKE_BUILD_TYPE=Release \
  -DWarpX_COMPUTE=CUDA \
  -DWarpX_DIMS=3 \
  -DWarpX_FFT=ON \
  -DWarpX_MPI=ON \
  -DWarpX_OPENPMD=ON \
  -DWarpX_QED=ON \
  -DWarpX_PYTHON=ON \
  -DPython_EXECUTABLE="$(which python3)"
cmake --build build -j 32
```
Also, we require to use QED table not downloaded in the previous orders, then copy the following files from this Github into the created warpx directory:

```bash
bw_table_chi_min_1.e-2_chi_max_1.e1_points_1024
qs_table_chi_min_1.e-5_chi_max_1.e1_points_1024
```
Additionally, download the following sbatch file to run jobs in subMIT and change the INPUT path at your convinience
```bash
perlmutter_fccz_random.sbatch
```
Submit the jobs as
```bash
sbatch perlmutter_fccz_random.sbatch
```
As a test file we use one in which beams' momenta are not originally rotated with reduced grid number to $256^3$:
```bash
Formenti.in
```
