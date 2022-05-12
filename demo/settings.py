import numpy as np
import sys

map_index_func = {1: 'sine'}

# which programs to infer MRs from
func_indices = [1]

# which type of MRs to infer: NOI_MIR_MOR_DIR_DOR.
# NOI: number of involved inputs
# MIR, MOR: mode of input and output relations. 1-equal, 2-greaterthan, 3-lessthan
# DIR, DOR: degrees of input and output relations. 1-linear, 2-quadratic, etc.
parameters_collection = ["2_1_1_1_1"]

# program to infer MR from
# user can encapsulate his program under test here and assign it a func_index, which can be used to faciliate processing for multiple programs in a batch
# i is the array containing the input arguments (note that unary input is treated as a special case of multi-variate input)
# o is the array containing the returning results (note that unary output is treated as a special case of multi-variate output)
# func_index is assigned to facilitate batch processing for multiple programs
def program(i, func_index):
    if func_index == 1:
        o = np.sin(i)
    else:
        pass
    return o


# the number of elements of the input for the programs
def getNEI(func_index):
    if func_index == 1:
        return 1
    else:
        pass

    
# domain for each element of the input
def get_input_range(func_index):
    if func_index == 1:
        return [[0, 20]]
    else:
        pass

# datatype for each element of the input:
def get_input_datatype(func_index):
    if func_index == 1:
        return [float]
    else:
        pass
    
# the number of elements of the output for the programs
def getNEO(func_index):
    if func_index == 1:
        return 1
    else:
        pass



# path to store results
output_path = "./output/demo"

# search parameters
pso_runs = 3
pso_iterations = 350





# set type and search range for coeff_range, const_range
coeff_type = int
const_type = float
coeff_range = [-2, 2]
const_range = [-10, 10]
