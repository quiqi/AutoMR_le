# 这是c语言版本的setting，原来的版本在setting1中

import numpy as np
import sys
import fdlibm

# QUQI：map_index_func中有两个为 15 的 key 值，修正之。
# map_index_func = {1:'abs', 2:'arcvos', 3:'arccosh', 4:'arcsin', 5:'arcsinh', 6:'arctan', 7:'arctan2', 8:'arctanh', 9:'ceil', 10:'cos', 11:'cosh', 12:'exp', 13:'floor', 14:'hypot', 15:'log', 15:'log1p', 17:'log10', 18:'amax', 19:'amin', 20:'round', 21:'sin', 22:'sinh', 23:'sqrt', 24:'tan', 25:'tanh'}
map_index_func = {1:'abs', 2:'arcvos', 3:'arccosh', 4:'arcsin', 5:'arcsinh', 6:'arctan', 7:'arctan2', 8:'arctanh',
                  9:'ceil', 10:'cos', 11:'cosh', 12:'exp', 13:'floor', 14:'hypot', 15:'log', 16:'log1p', 17:'log10',
                  18:'amax', 19:'amin', 20:'round', 21:'sin', 22:'sinh', 23:'sqrt', 24:'tan', 25:'tanh'}

# which programs to infer MRs from
func_indices = list(range(1, 26))

# which type of MRs to infer: NOI_MIR_MOR_DIR_DOR.
# NOI: number of involved inputs
# MIR, MOR: mode of input and output relations. 1-equal, 2-greaterthan, 3-lessthan
# DIR, DOR: degrees of input and output relations. 1-linear, 2-quadratic, etc.
parameters_collection = ["2_1_1_1_1", "2_1_1_1_2", "2_1_1_1_3", "3_1_1_1_1", "3_1_1_1_2", "2_1_2_1_1", "2_1_3_1_1", "2_2_1_1_1", "2_3_1_1_1", "2_2_2_1_1", "2_2_3_1_1", "2_3_2_1_1", "2_3_3_1_1"]


# 读入c语言编写
cmath = fdlibm.Interface('./doc/put.so')

# program to infer MR from
# user can encapsulate his program under test here and assign it a func_index, which can be used to faciliate processing for multiple programs in a batch
# i is the array containing the input arguments (note that unary input is treated as a special case of multi-variate input)
# o is the array containing the returning results (note that unary output is treated as a special case of multi-variate output)
# func_index is assigned to facilitate batch processing for multiple programs
def program(i, func_index):
    o = []
    if func_index == 1:
        for t in i:
            o.append(cmath.fabs(t))
    elif func_index == 2:
        for t in i:
            o.append(cmath.acos(t))
    elif func_index == 3:
        for t in i:
            o.append(cmath.acosh(t))
    elif func_index == 4:
        for t in i:
            o.append(cmath.asin(t))
    elif func_index == 5:
        for t in i:
            o.append(cmath.asinh(t))
    elif func_index == 6:
        for t in i:
            o.append(cmath.atan(t))
    elif func_index == 7:
        for (t0, t1) in zip(i[0], i[1]):
            o.append(cmath.atan2(t0, t1))
    elif func_index == 8:
        for t in i:
            o.append(cmath.atanh(t))
    elif func_index == 9:
        for t in i:
            o.append(cmath.ceil(t))
    elif func_index == 10:
        for t in i:
            o.append(cmath.cos(t))
    elif func_index == 11:
        for t in i:
            o.append(cmath.cosh(t))
    elif func_index == 12:
        for t in i:
            o.append(cmath.exp(t))
    elif func_index == 13:
        for t in i:
            o.append(cmath.floor(t))
    elif func_index == 14:
        for (t0, t1) in zip(i[0], i[1]):
            o.append(cmath.hypot(t0, t1))
    elif func_index == 15:
        for t in i:
            o.append(cmath.log(t))
    elif func_index == 16:
        for t in i:
            o.append(cmath.log1p(t))
    elif func_index == 17:
        for t in i:
            o.append(cmath.log10(t))
    elif func_index == 18:
        o = np.amax(i)
    elif func_index == 19:
        o = np.amin(i)
    elif func_index == 20:
        o = np.round(i)
    elif func_index == 21:
        for t in i:
            o.append(cmath.sin(t))
    elif func_index == 22:
        for t in i:
            o.append(cmath.sinh(t))
    elif func_index == 23:
        for t in i:
            o.append(cmath.sqrt(t))
    elif func_index == 24:
        for t in i:
            o.append(cmath.tan(t))
    elif func_index == 25:
        for t in i:
            o.append(cmath.tan(t))
    else:
        pass
    o = np.array(o)
    return o


# the number of elements of the input for the programs
def getNEI(func_index):
    if func_index in [7, 14, 18, 19]:
        return 2
    else:
        return 1

# the number of elements of the output for the programs
def getNEO(func_index):
    return 1

# domain for each element of the input
# QUQI: 修正函数取值范围
def get_input_range(func_index):
    if func_index in [7, 14, 18, 19]:
        return [[0, 20], [0, 20]]
    elif func_index in [15, 17]:    # log, log10函数输入需大于0。
        return [[0, 20]]
    elif func_index in [2,4]:   # arccos, arcsin定义域在[-1,1]之间
        return [[-1, 1]]
    elif func_index in [3]:     # arccosh定义域在[1,+∞]。
        return [[1, 20]]
    elif func_index in [16]:    # log(1+x)的定义域在[-1,+∞]。
        return [[-1, 20]]
    else:
        return [[-10, 10]]


# datatype for each element of the input:
def get_input_datatype(func_index):
    if func_index in [7, 14, 18, 19]:
        return [float, float]
    else:
        return [float]



# path to store results
output_path = "./output/np25"

# search parameters
pso_runs = 3
pso_iterations = 350




# set type and search range for coeff_range, const_range
coeff_type = int
const_type = float
coeff_range = [-2, 2]
const_range = [-10, 10]