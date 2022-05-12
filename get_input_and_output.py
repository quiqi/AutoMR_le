"""
author: QUQI
tel: 13142338890
email: 1292396438@qq.com
"""
import csv
import os.path

import numpy as np


def get_i1(str_r, x, g=None):
    """
    通过str_r描述的输入关系，根据x产生第二次输入的值，返回一个基于str_r运算得到的与x等长的向量
    :param str_r: 输入关系
    :param x: 第一次输入
    :param g: str_r中各参数的含义，如果str描述为初等函数，且无未知参数，可无需传入
    :return:
    """
    gn = {'np': np, 'x': x}
    if g is not None:
        for i in g:
            gn[i] = g[i]
    inputs = []
    for r in str_r:
        exec('i1=' + r, gn)
        inputs.append(gn['i1'])
    return np.array(inputs)


def get_x(scope, num=500):
    """
    根据scope定义的取值范围范围随机产生待测试值
    :param scope: 取值范围参数，若scope为一维list，如：[-10,10]，则返回一个长度为num的一维随机数组；
    若为二维1list，如：[[-10,10],[-10,10]],则返回一个二维的数组，大小为：len(scope)*num
    :param num: 返回数组大小，默认为 500
    :return: 生成好的随机数组x
    """
    if num < 0:
        raise Exception('请保证num>0')
    scope = np.array(scope)
    if len(scope.shape) == 1:   # 如果scope是一个一维数组，则返回一个一维x值
        if len(scope) != 2:
            raise Exception('get_x的scope参数需要是一个长度为2的数组或一组长度为2的数组')
        elif scope[0] > scope[1]:
            raise Exception('给定值域的起点需要小于终点')
        return np.random.random_sample(num)*(scope[1]-scope[0]) + scope[0]
    elif len(scope.shape) == 2:
        xs = [get_x(s, num) for s in scope]
        return np.array(xs)
    else:
        return Exception('get_x的scope参数需要是一个长度为2的数组或一组长度为2的数组')


def creat_io(scope, rs, fun, save_path, g=None, num=3):
    input0 = get_x(scope, num)
    rs = np.array(rs)
    if len(rs.shape) == len(input0.shape) == 1:  # 如果是一维
        inputs = [get_i1([r], input0, g) for r in rs]
        input0 = [input0]
    elif len(rs.shape) == len(input0.shape) == 2:   # 如果是多维输入
        inputs = [get_i1(r, input0, g) for r in rs]
    else:
        raise Exception('请确保输入关系的正确性')

    output0 = fun(*tuple(input0))
    outputs = []
    for i in inputs:
        outputs.append(fun(*tuple(i)))
    outputs = np.array(outputs)

    np.savez(save_path, I_0=input0, I_S=inputs, O_0 = output0, O_S = outputs)


def npz_to_csv(x):
    csv_head = []
    block = []
    for i in range(x['I_0'].shape[0]):
        csv_head.append('I_(0,{})'.format(i))
        block.append(x['I_0'][i])
    for i in range(x['I_S'].shape[0]):
        for j in range(x['I_S'].shape[1]):
            csv_head.append('I_({},{})'.format(i+1, j))
            block.append(x['I_S'][i][j])
    csv_head.append('O_0')
    block.append(x['O_0'])
    for i in range(x['O_S'].shape[0]):
        csv_head.append('O_{}'.format(i+1))
        block.append(x['O_S'][i])
    block = np.array(block).transpose()
    return csv_head, block


if __name__ == '__main__':
    # 此脚本用于生成输入输出，此脚本不与任何其他文件关联，可单独使用
    funs = {    # 修改funs可以生成自定义数据
        # 函数名：(函数, [i1=f(i0), i2=f(i0)], 函数输入取值范围)
        'sin': (np.sin, ['x+3.14', '-x'], [-10, 10]),
        'arcsin': (np.arcsin, ['-x'], [-1, 1]),
        'arcsinh': (np.arcsinh, ['-x'], [-10, 10]),
        'log': (np.log, ['2*x'], [0, 10]),
        'exp': (np.exp, ['2*x'], [-10, 10]),
        'hypot': (np.hypot, [['x[1]', 'x[0]']], [[-10, 10], [-10, 10]])
    }
    for name in funs:
        root = './output_npz_csv/{}/'.format(name)
        if not os.path.isdir(root):
            os.makedirs(root)
        for i in range(100):
            creat_io(funs[name][2], funs[name][1], funs[name][0], root + '{}_{}.npz'.format(name, i + 1), num=100)

    for name in funs:
        with open('./output_npz_csv/{}.csv'.format(name), 'w', newline='') as cfa:
            cwa = csv.writer(cfa)
            k = 0
            root = './output_npz_csv/{}/'.format(name)
            for i in range(100):
                x = np.load('./output_npz_csv/{}/{}_{}.npz'.format(name, name, i+1))
                head, block = npz_to_csv(x)
                with open('./output_npz_csv/{}/{}_{}_{}.csv'.format(name, str(funs[name][1]).replace('*', ''), name, i+1), 'w', newline='') as cf:
                    cw = csv.writer(cf)
                    cw.writerow(head)
                    if k == 0:
                        k = 1
                        cwa.writerow(head)
                    cw.writerows(block)
                    cwa.writerows(block)