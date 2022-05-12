# AutoMR
A tool to discover polynomial metamorphic relations (MRs).

_A Simple Example:_ To calculate ___sin(x)___, the program can be implemented as series:
```
sin(x) ≈ x - (x^3)/3! + (x^5)/5! - (x^7)/7! ...
```

It's difficult to find some properties of this program such as its relation with 3.1415926 (pi). However, AutoMR could find such interesting MRs: 
- sin(x) + sin(-x) = 0
- sin(x) + sin(x + 3.14159265) ≈ 0
- sin(x) - sin(x + 6.28318531) ≈ 0
- ...

_Another Example:_ For the program ___max(*args)___, AutoMR could find such MRs: 
- ___max(i<sup>(1)</sup>, i<sup>(2)</sup>, i<sup>(3)</sup>) - max(i<sup>(2)</sup>, i<sup>(1)</sup>, i<sup>(3)</sup>) = 0___
- ___max(i<sup>(1)</sup>-1, i<sup>(2)</sup>-2, i<sup>(3)</sup>-2) - max(i<sup>(1)</sup>, i<sup>(2)</sup>, i<sup>(3)</sup>) < 0___
- ...

## How to use AutoMR

### 0. Environemt settings
Tested on Windows 10 1809, Ubuntu 18.04, macOS Mojave with the following packages:
* python 3.6.3
* numpy 1.14.5
* pandas 0.23.1
* sympy 1.1.1
* scipy 1.1.0
* z3-solver 4.8.5.0
* scikit-learn 0.19.2

[Anaconda](https://www.anaconda.com/what-is-anaconda) is recommended to set up the environment.
With Anaconda, the dependencies can be easily installed by: 
```aidl
>>> cd path/to/AutoMR
>>> conda env create -f environment.yml
```

A virtual envoronement named "AutoMR" will be created together with the required dependencies. The following cmmand will activate the virtual environment named "AutoMR":
```
>>> conda activate AutoMR
```


### 1. Subject Programs

The information of the program which you want to infer MRs from should be configured in _settings.py_
> 1. Encapsulate the program in _program(i, func_index)_. _i_ is an array containing all the values to be passed to the program. _func_index_ is the index assigned to the program, which can facilitate inferring MRs for various programs in a batch.
> 2. Provide the number of elements of the input of the program in _getNEI(func_index)_.
> 3. Provide the number of elements of the output of the program in _getNEO(func_index)_.
> 4. Provide the input domain in _get_input_range(func_index)_.
> 5. Provide the input data type in _get_input_datatype(func_index)_.
> 6. Set the type of MRs you want to infer in _parameters_collection_. Each type is represented by a string consisting of "NOI_MIR_MOR_DIR_DOR". NOI is number of inputs involved in the target MR. MIR is mode of the input relation. MOR is mode of the output relation. For MIR and MOR, 1 means equality, 2 means greater-than, 3 means less-than. DIR and DOR is the polynomial degree of the input and output relations: 1 is linear, 2 is quadratic, etc.
> 7. Set the path to store the results in _output_path_.
> 8. Set the number of searches in _pso_runs_.


### 2. Search for MRs (Phase 1), filter the search results (Phase 2) and remove redundant MRs (Phase 3)
After setting up the subject programs and parameters, execute the following command. The results of each phase will be stored in corresponding folder under the _output_path_.
```
>>> python main.py
```

## A minimal example
The ___settings.py___ has already set up for inferring MRs for __sine__ program. You can just clone this repo and run `python main.py`, then after searching the results will be shown.

The following code block shows an example run and the results. The results can also be viewed as html file in the output folder.
```
>>> python main.py
=====Start Inferring=====
start phase 1: searching ...
start phase 2: filtering ...
start phase 3: redundancy removing ...
=====Results=====
result file is 21_MRs_other_types_after_cs_svd.pkl
----------
func_index is 21, NOI_MIR_MOR_DIR_DOR is 2_1_1_1_1:
             1  prod(i0_1,)
i1_1 -3.141593         -1.0
i2_1  3.141593         -1.0
i3_1 -9.424778         -1.0
i4_1  3.141593          1.0
i5_1 -9.424778          1.0
                1   prod(o0_1,)  prod(o1_1,)  prod(o2_1,)  prod(o3_1,)  prod(o4_1,)  prod(o5_1,)
MR1 -7.620898e-16  2.236068e+00    -0.447214    -0.447214    -0.447214     0.447214     0.447214
MR2  1.005613e-16  2.359224e-16     0.000000    -0.206041     0.755483     0.618123    -0.068680
MR3  1.613963e-16 -1.110223e-16     0.000000     0.135266     0.384900    -0.330794     0.850959
MR4  1.067449e-16  2.220446e-16     0.000000     0.830211    -0.176345     0.508429     0.145437

```

### Associated Paper
The folder "Associated_research_paper" contain the research paper manuscript and the experimental data (results of inferred MRs from a number of NumPy and Apache Math programs).
