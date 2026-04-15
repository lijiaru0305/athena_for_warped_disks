# Athena++ for Warped Disks

This is my personal fork of [Athena++](https://github.com/PrincetonUniversity/athena) for 3D hydrodynamic simulations of **warped protoplanetary disks**.

## What this repository does

This code is designed to study:

- steady-state warp structures
- nonlinear hydrodynamics of warped disks
- disk breaking

The main setup fixes the disk inclination at the inner and outer radial boundaries, allowing the gas disk to develop a steady warp structure self-consistently without specifying an explicit external perturber.

## Main additions relative to upstream Athena++

- warped-disk problem generators (other pgens are removed)
- tilted initial conditions
- fixed-tilt radial boundary conditions

## Build

```bash
module load intel/2024.0
module load mpi/intel-mpi-5.1.3.258
module load hdf5/1.10.7-openmpi-intel-2021.4.0

python configure.py --prob=<problem_name> --coord=spherical_polar --eos=isothermal --flux=llf -hdf5 -h5double -mpi --hdf5_path=<hdf5_path> -omp --cflag=-qno-openmp-simd
make -j
```

## Run

``` bash
./bin/athena -i inputs/<input_file>.athinput
```
or with MPI:
```bash
mpirun -np 16 ./bin/athena -i inputs/<input_file>.athinput
```

## Upstream code
This is a fork of [Athena++](https://github.com/PrincetonUniversity/athena) ; please also cite:
```
@article{Stone2020,
	doi = {10.3847/1538-4365/ab929b},
	url = {https://doi.org/10.3847%2F1538-4365%2Fab929b},
	year = 2020,
	month = jun,
	publisher = {American Astronomical Society},
	volume = {249},
	number = {1},
	pages = {4},
	author = {James M. Stone and Kengo Tomida and Christopher J. White and Kyle G. Felker},
	title = {The Athena$\mathplus$$\mathplus$ Adaptive Mesh Refinement Framework: Design and Magnetohydrodynamic Solvers},
	journal = {The Astrophysical Journal Supplement Series},
}
```



