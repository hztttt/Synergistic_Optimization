# Synergistic optimization

---------------------------------------------------------------------

This repository provides access to perform synergistic optimization via trained [DeepSPIN](https://doi.org/10.48550/arXiv.2304.09606) model interfaced with modified [LAMMPS](https://github.com/lammps/lammps/tree/stable_23Jun2022) code.

## Contents

- [Requirements](#requirements)
- [Dataset for FCC Fe](#dataset-for-fcc-fe)
- [Perform synergistic optimization](#perform-synergistic-optimization)
- [Train DeepSPIN model](#train-deepspin-model)
    - [Dataset preparation](#dataset-preparation)
    - [Input script preparation](#input-script-preparation)
    - [Train a model](#train-a-model)
    - [Freeze a model](#freeze-a-model)


## Requirements

### Install DeepSPIN

In DeepSPIN method, Python 3.7 or later environments and following packages are required：

- deepmd-kit>=2.2.2
- dpdata

To apply synergistic optimization, a manual installation of DeePMD-kit is required from source codes by following the instructions [installing the Python interface](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/install/install-from-source.md#install-the-python-interface) and [installing the C++ interface](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/install/install-from-source.md#install-the-c-interface), since the C++ interface is necessary when using DeePMD-kit together with LAMMPS. 

For a simple usage of DeepSPIN, you can easily achieve the requirements and install DeepSPIN with `conda`,

```shell
# install miniconda with python 3.9
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh

# install deepmd-kit packages with conda 
conda create -n deepmd deepmd-kit=2.2.2 -c conda-forge

# enable the environment
conda activate deepmd

# install dpdata packages with pip
pip install dpdata
```

### Install modified LAMMPS

After DeepSPIN configuration, then you need to make LAMMPS interface.

```shell
cd $deepmd_source_dir/source/build
make lammps
```

It will generate a module called `USER-DEEPMD` in the `build` directory.

We modified the source codes of LAMMPS to achieve the synergistic optimization based on the version `stable_23Jun2022_update1`, as shown in `lammps_DeepSPIN`. You may download it, copy the module `USER-DEEPMD`, and compile necessary modules as follows. 

```shell
cd lammps_DeepSPIN/src
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
make yes-kspace
make yes-manybody
make yes-spin
make yes-user-deepmd
```

Then build LAMMPS and end up with an executable `lmp_mpi` as follows,

```shell
make mpi -j 4
```
More details about the modifications are provided in [minimization algorithm](lammps_DeepSPIN/src/SPIN/min_spin_cg.cpp).

## Dataset for FCC Fe

The dataset for FCC Fe of perfect and defective structures is provided in `dataset/FCC_Fe` and `dataset/FCC_Fe_defect`. More information about the dataset preparation for DeepSPIN training is detailed in [Dataset preparation](#dataset-preparation).

## Perform synergistic optimization

 Equipped with DeepSPIN model and LAMMPS, We provide the input and output files of running the synergistic optimization in `example/synergistic_optimization`. `init.data` specifies the initial lattice and spin configuration of perturbed FCC Fe with 4000 atoms. For the detailed introduction of data file format please refer to this [documentation](https://docs.lammps.org/read_data.html). `model.pb` is the trained DeepSPIN model. `minimize.in` is the input script to run LAMMPS. You can simply run the optimization by mpirun,

```shell
mpirun -np 1 lammps_DeepSPIN/src/lmp_mpi -in minimize.in
```

You can change `-np` to the number of processes suitable to your own machine.

When the optimization is finished, a series of output files will be generated in the current directory. Especially, `*.out` records the optimized spin configuration of the system, in which you may see a deviation of the spin configuration from the AFMD ground state.

## Train DeepSPIN model

The usage of DeepSPIN is similar to that of [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit). The guide of a quick start on DeePMD-kit can be found [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/getting-started/quick_start.ipynb). In this part we show an example of how to train a DeepSPIN model of FCC Fe system. 

### Dataset preparation

The original training data should be sourced from the first-principle non-collinear magnetic excitation calculation method [DeltaSpin](https://github.com/caizefeng/DeltaSpin) based on VASP, in which the lattice configuration is collected from `POSCAR`, the atomistic spin configuration is collected from `INCAR`, the potential energy of system and atomic forces are collected from `OUTCAR`, and magnetic forces are collected from `OSZICAR`. We take the 32-atom $2\times2\times2$ FCC Fe supercell for the calculation and collect DFT results. 

After some preprocessing, the format of original data should be converted to fit the general input file requirements for DeePMD-kit, as shown in `example/Fe/raw`, more detailed description is given in this [documentation](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/data/index.md). It's worth noting that the type of pseudo atoms around Fe atoms is represented by `1` in `type.raw`, the Cartesian coordinates of pseudo atoms are stitched after the coordinates of real atoms sequentially by index in `coord.raw`, and the magnetic forces of Fe atoms are stitched after the atomic forces of real atoms sequentially as well in `force.raw`. 

Then the data should be split into training set and validation set, converted from `.raw` format to `.npy` format using the script `raw_to_set.sh` in `example/Fe/scripts`. For example, we provide 100 frames of data in `example/Fe/raw`,

```shell
cd example/Fe/raw
sh ../scripts/raw_to_set.sh 20
```

It will generate 5 subsets from `set.000` to `set.004` in the current directory, with each subset containing 20 frames. The former 4 subsets can be picked as the training set, while the last one will be the validation set, as shown in `example/Fe/data`. 

### Input script preparation

Once the dataset preparation is complete, a `json` format input script is required to specify the parameters for model training. Here we take `example/Fe/Fe.json` as an example, and the parameters dedicated to DeepSPIN are introduced. For detailed information about other parameters, please refer to this [documentation](https://github.com/deepmodeling/deepmd-kit/tree/master/doc/model).

The hyper-parameters dedicated to DeepSPIN is set in the following section

```json
    "spin" : {
        "use_spin":         [true],
        "virtual_len":      [0.3],
        "spin_norm":        [1.323]
    },
```
* `use_spin` determines whether the atom type is magnetic. Here We set `true` for Fe.
* `virtual_len` specifies the distance between the pseudo atom and its corresponding real atom. Here we set 0.3 Å for Fe.
* `spin_norm` specifies the magnitude of the magnetic moment for each magnatic atom. Here we set 1.323 $\mu_B$ for Fe.

The loss function for the DeepSPIN model is set in the following section

```json
    "loss" : {
        "type":               "ener_spin",
        "start_pref_e":       0.02,
        "limit_pref_e":       200,
        "start_pref_fr":      1000,
        "limit_pref_fr":      1,
        "start_pref_fm":      10000,
        "limit_pref_fm":      10
    },
```

where `start_pref_e`, `limit_pref_e`, `start_pref_fr`, `limit_pref_fr`, `start_pref_fm` and `limit_pref_fm` determines the starting and ending weight of energy, atomic forces and magnetic forces in the loss function respectively.

### Train a model

The training of a model can be simply executed by running

```shell
cd example/Fe/train
dp train Fe.json
```

If the training process is successfully running, a series of files will be generated in the current directory. The learning curve on both training set and validation set can be viewed from `lcurve.out`. For more details please refer to this [documentation](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/train/training.md).

### Freeze a model

When the training is finished, the DeepSPIN model can be extracted from a checkpoint and dumped into a protobuf file `*.pb`. This process is called "freezing" by running

```shell
dp freeze -o model.pb
```

Then you will obtain the frozen DeepSPIN model `model.pb`.