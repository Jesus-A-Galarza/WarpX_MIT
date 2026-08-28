# Plotting scripts
## plots_base
This reads (by default) the diags directory produced by WarpX and produces two directories: diags_iteration_analysis and diags_iteration_analysis_corrected.

In WarpX, if a particle is located inside the simulation grid at a certain iteration but it is calculated to be outside of the grid for the next iteration then warpx will record its position outside of the grid and this information will be plotted under diags_iteration_analysis.

Using a simple interpolation to calculate the trajectory of a particle outside of the grid, diags_iteration_analysis_corrected contains the information of the particle to the calculated location of the grid.

That is, for particles outside of the simulation, it traces them back to where they would have scapped the grid and produces the plots based on that information.

The directories used and graphs produced are:

particles_in, particles_out, and combined.

The physics graphs produced in both directories are:

spatial distributions and overlays, 2-D graphs for spatial distributions, phase space ($log_{10}(\theta) \, v. log_{10}(pT)$), luminosity, pair multiplicity, and fraction percentages.

The particles read are:
beam1(electrons), beam2(positrons), (ele & pos)_bw, (ele & pos)_bh, and (ele & pos)_bw.
Other information that this script reads is the location for when the IPC's are produced.
### plots_base commands
```bash
--diags PATH
```
WarpX will create a directory by default in which it stores the information for particles_in and particles_out but if you desire to change the input file you can just modify the PATH argument to the desired directory.
```bash
--corrected-out PATH
```
Similar as the previous one but for the corrected directory.
```bash
--label TEXT
```
Gives a label to the plots.
```bash
--iterations [N ...]
```
Restricts the analysis to specific $available$ iterations.
```bash
--outside-mode MODE
```
batch or cumulative. For example, in analysis for iteration 100, batch-mode will read particles_in and particels_out only in iteration 100 (by WarpX ScrapBoudary function in the grid, it will record all particles that reach the grid, whether in previous iterations or not, then this is the desired and default mode) but cumulative will read only particles_in in iteration 100 and particles_out for all iteration less than and equal to 100.

Magenta lines in the plots mark the simulation limits and these must be hard-coded in the plotting script if you chose to use different box dimensions but these do not affect the simulation or plotting process with the exception of production particles.
# Fix Position and Momentum

## corrected_boosted
'fix_position_momentum' reads the `diags` directory produced by WarpX and creates a complete copied directory called `diags_corrected_boosted`. The original `diags` directory is not modified.
The first correction is for particles recorded by the WarpX BoundaryScraping diagnostics. In WarpX, if a particle is inside of the simulation grid at one iteration but is calculated to be outside of the grid at the next iteration, WarpX can record the particle at its calculated position outside of the simulation box.
For particles in the `particles_out` diagnostics, the script uses the recorded position and the original propagated momentum of the particle to trace its trajectory backwards. It calculates the intersection between this trajectory and the simulation box and replaces the outside position with the location where the particle would have crossed the simulation boundary. Particles that are already inside the grid or already located on a boundary are left unchanged. If an intersection cannot be calculated, the original position is also left unchanged. 
The simulation box dimensions used for this correction are:
x = 2083.55 micrometres
y = 26.8711 micrometres
z = 267.170 millimetres
These dimensions are hard coded in the script. The second correction applies the crossing-angle Lorentz boost to the IPC particles.
The script uses a crossing angle of 30 mrad and performs a common boost in the positive x direction using half of the crossing angle. For the propagated particle momentum, the Lorentz transformation is applied to the x momentum while the y and z momentum components remain unchanged. The particle energy is calculated using the full momentum and electron mass before applying the boost. The script also applies the crossing-angle transformation to the momentum stored at the location where the IPC was originally produced. These quantities are stored as `origUx`, `origUy`, and `origUz`, which represent the proper velocity $\gamma v$. The script converts these values into physical momentum using
