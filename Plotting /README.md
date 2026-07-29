# Plotting scrips
## plots_base
This reads (by default) the diags directory produced by WarpX and produces two directories: diags_iteration_analysis and diags_iteration_analysis_corrected.
In WarpX, if a particle is located inside the simulation grid at a certain iteration but it is calculated to be outside of the grid for the next iteration then warpx will record its position outside of the grid and this information is saved under diags_iteration_analysis.
Using a simple interpolation to calculate the trajectory of a particle outside of the grid, diags_iteration_analysis_corrected contains the infomration of the particle to the calculated location of the grid. That is, for particles outside of the simulation, it traces them back to where they would have scapped the grid and produces the plots based on that information.
The graphs produced in both directories are:
spatial distributions and overlays, 2-D graphs for spatial distributions, phase space ($log_{10}(\theta) \, v. log_{10}(pT)$
### plots_base commands
```bash
--diags PATH
```
WarpX will create a directory by default in which it stores the information for particles_in and particles_out but if you desire to change the input file you can just modify the PATH argument to the desired directory.
```bash
```
