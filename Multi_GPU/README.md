# Running WarpX with Multiple GPUs
## 1 GPU configuration limitations
Running WarpX with a number of cells greater than $576^3$ requires a memory higher than 24Gb while subMIT nodes have a capacity of 21 Gb. Then, using multiple GPU's becomes imperative.

## 2 GPU configuration
While it is possible to run with multiple GPU's with MPI, subMIT capacity is shared among all their users and, consequently, availability for a number of GPUs higher than 2 is scarce. The recommendation is to send an email to subMIT team for a reserved daily usage if you would like to run complex jobs. The following lines are the information of the nodes and their addresses. These are easily replaceable in the batch job to point towards an specific node in the batch file:

```bash
---- nvidia_a30
submit20-1 gpu:2
submit21-1 gpu:2
submit22-1 gpu:2
submit23-1 gpu:2

---- Tesla_v100
submit77-1 gpu:4

---- nvidia_rtx6000
submit37 gpu:2

---- gtx1080
submit60to73 gtx1080
```
Acknowledging these limitations, modifications in both the warpX input file and in the batch are required to run with (in this case) 2 GPUs. For this case we included the line in the warpX input:
```bash
warpx.numprocs = 1 1 2
```
This line controls the MPI domain decomposition. It tells warpX to divide the z-axis into 2 sections, creating two computational boxes, one for each MPI rank.

Then you can run
```bash
sbatch perlmutter_fccz_random.sbatch
```
to obtain your job
### WarpX file modification to allow multiple GPUs usage
#### Atention
WarpX required the product of warpx.numprocs to equal the number of MPI ranks, one MPI per GPU.
#### Increase the Grid while keeping 2 GPU
For this change you only need to modify the warpX input file, but, as mentioned before, these may induce computational problems due to the memory usage.
#### Increasing the number of GPUs
For 
```bash
my_constants.nx = N
my_constants.ny = N
my_constants.nz = N

warpx.numprocs = 1 1 G
```
we require to modify the sbatch file to 
```bash
#SBATCH --ntasks=G
#SBATCH --gres=gpu:G
```
However, for this purpose one has to be sure that the current node possesses the required number of GPUs. One can check that with the following command
```bash
scontrol show node submit37 | grep -E "Gres=|CfgTRES="
```
in which we use submit37 as an example, you can modify this part to check any node.
