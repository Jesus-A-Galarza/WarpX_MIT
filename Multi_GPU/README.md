# Running WarpX with Multiple GPUs
## 1 GPU configuration limitations
Running WarpX with a number of cells greater than $576^3$ requires a memory higher than 24Gb while subMIT nodes have a capacity of 21 Gb. Then, using multiple GPU's becomes imperative; however, there are still limitations. 

## 2 GPU configuration
While it is possible to run with multiple GPU's with MPI, subMIT capacity is shared among all their users and consequently availability for a number of GPUs higher than 2 is scarce. The recommendation is to send an email to subMIT team for a reserved daily usage if you would like to run complex jobs. The following lines are the information of the nodes and their addresses. These are easily replaceable in the batch job to point towards an specific node in the batch file:

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
Acknowledging these limitations, modifications in both the warpX input file and in the batch are required to run with (in this case) 2 GPUs. Warp
### WarpX file modification to allow multiple GPUs usage
\

