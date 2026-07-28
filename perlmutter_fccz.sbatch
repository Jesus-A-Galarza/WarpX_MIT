#!/bin/bash
#SBATCH --job-name=warpx
#SBATCH --output=WarpX_%j.txt
#SBATCH --error=WarpX_%j.txt
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -x

echo "========== JOB INFO =========="
echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo

echo "========== SOURCING CUDA ENVIRONMENT =========="
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
echo "LCG CUDA setup status: $?"

source /cvmfs/sft.cern.ch/lcg/releases/LCG_108_cuda/hdf5_mpi/1.14.6/x86_64-el9-gcc13-opt/hdf5_mpi-env.sh
echo "HDF5 MPI setup status: $?"
echo

export CUDA_ROOT=/cvmfs/sft.cern.ch/lcg/releases/cuda/12.5-3429b/x86_64-el9-gcc13-opt
export CUDA_HOME=${CUDA_ROOT}

export PATH=${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}

echo "========== SLURM / CUDA INFO =========="
echo "CUDA_ROOT=${CUDA_ROOT}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo

echo "========== GPU INFO =========="
which nvidia-smi || true
nvidia-smi || true
echo

echo "========== CUDA PATH =========="
which nvcc || true
nvcc --version || true
echo

echo "========== MPI INFO =========="
which mpirun || true
mpirun --version || true
echo

echo "========== CUDA LIBRARY CHECK =========="
echo "Checking CUDA libraries in CUDA_ROOT:"
ls -lh ${CUDA_ROOT}/lib64/libcufft* || true
echo

EXE=/work/submit/jesusgal/FCCPhysics/Summer/GPU_2/warpx/build/bin/warpx.3d
INPUTS=/work/submit/jesusgal/FCCPhysics/Summer/GPU_2/warpx/Formenti.in

echo "========== EXECUTABLE CHECK =========="
echo "EXE=${EXE}"
ls -lh "${EXE}"
EXE_STATUS=$?
echo

echo "========== INPUT FILE CHECK =========="
echo "INPUTS=${INPUTS}"
ls -lh "${INPUTS}"
INPUT_STATUS=$?
echo

if [ ${EXE_STATUS} -ne 0 ]; then
    echo "ERROR: WarpX executable was not found or is not accessible."
    exit 10
fi

if [ ${INPUT_STATUS} -ne 0 ]; then
    echo "ERROR: WarpX input file was not found or is not accessible."
    exit 11
fi

echo "========== LINKED CUDA LIBRARIES =========="
ldd "${EXE}" | grep -Ei "cuda|cufft|cudart" || true
echo

echo "========== MISSING LIBRARY CHECK =========="
ldd "${EXE}" | grep "not found" || true
MISSING_LIBS=$(ldd "${EXE}" | grep "not found" || true)

if [ -n "${MISSING_LIBS}" ]; then
    echo "ERROR: Missing shared libraries detected:"
    echo "${MISSING_LIBS}"
    exit 12
fi
echo

echo "========== INPUT FILE PREVIEW =========="
head -n 80 "${INPUTS}"
echo

echo "========== STARTING WARPX =========="
echo "Command:"
echo "mpirun -np 1 ${EXE} ${INPUTS}"
echo

mpirun -np 1 "${EXE}" "${INPUTS}" 2>&1 | tee output_${SLURM_JOB_ID}.txt
WARPX_STATUS=${PIPESTATUS[0]}

echo
echo "========== WARPX FINISHED =========="
echo "WarpX exit code: ${WARPX_STATUS}"
echo "End time: $(date)"
echo

echo "========== GPU INFO AFTER RUN =========="
nvidia-smi || true
echo

exit ${WARPX_STATUS}
