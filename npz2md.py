"""
author: QUQI
tel: 13142338890
email: 1292396438@qq.com
"""
from load_resulrs  import *
import os
import settings
import json


def column_to_latex(column, s: str):
    """
    将列名转化为latex形式
    :param column: 待转化的列名
    :param s: 变量名
    :return: 转化好的latex表示
    """
    items = []
    for c in column:
        if type(c) is tuple:
            pass
        item = ''
        if c != '1':  # 如果不是常数项,则将该列名转化为latex公式的形式
            xs = c[5:-1].replace(' ', '').split(',')     # 得到所有的一次项
            es = {}

            # 统计每个一次项的次数
            for x in xs:
                if len(x) == 0:
                    continue
                if x not in es.keys():
                    es[x] = 0
                es[x] += 1

            # 将数据转化为latex公式的形式
            for x in es:
                i, j = x.replace(' ', '')[1:].split('_')
                val = s + '_{' + i + ',' + j + '}'
                if es[x] == 1:
                    item = item + val
                else:
                    item = item + val + '^{}'.format(es[x])
        items.append(item)

    return items


def mat_to_polynomial(mat, item):
    pols = []
    for i in mat:
        pol = []
        for a, x in zip(i, item):
            a = round(a, 3)
            if abs(round(a) - a) < 0.001:
                a = round(a)
            if abs(a) <= 0.001:  # 如果系数是0，直接跳过该项
                continue
            elif len(x) == 0:   # 如果是常数项，直接加入系数
                # pass
                pol.append(str(a))
            elif abs(a - 1.0) <= 0.001: # 如果系数是1，省略系数
                pol.append(x)
            elif abs(a + 1.0) <= 0.001:
                pol.append('-' + x)
            else:   # 否则写成系数和项目相乘的方式
                pol.append(str(a)+x)
        pol = '+'.join(pol)
        pol = pol.replace('+-', '-')
        if len(pol) == 0:
            pol = '0'
        pols.append(pol)
    return pols


def json_to_md(json_path, md_path):
    with open(json_path, 'r') as f:
        MRs1 = json.load(f)
        MRs = {}
        for i in MRs1:
            MRs[int(i)] = MRs1[i]

    with open(md_path, 'a', encoding='utf-8') as fw:
        # 写注释：
        # fw.write('> mod:i-j-k表示，共有i次输入，输入模式最高阶数为j, 输出模式最高阶数为k\n')
        for id in sorted(MRs.keys()):
            if len(MRs[id]) == 0:
                continue

            txts = ['## {}. {}'.format(id, MRs[id][0]['name'])]  # 二级标题：函数id加函数名

            con_row = '{} & {} & {}\\\\\n'

            for i, MR in enumerate(MRs[id]):
                # 获得该MR的模式：
                ni = str(MR['num_involved_inputs'])  # 输入元数
                ideg = str(str(MR['input_degrees']))  # 输入关系的阶数
                odeg = str(MR['output_degrees'])  # 输出关系的阶数
                mod = '-'.join([ni, ideg, odeg])

                txts.append('### mod:{}'.format(mod))  # 以模式为三级标题

                # 1. 得到输入关系：
                pols_x1 = mat_to_polynomial(MR['IR'], MR['item_X'])  # 将矩阵转化为多项式
                # 将多项式按输入分组
                step = len(MR['item_X']) - 1
                if step < 1:
                    step = 1
                pols_x2 = [pols_x1[i:i + step] for i in range(0, len(pols_x1), step)]

                # 生成input-relation
                txts.append('**relation of input:**')  # 输入关系
                for i, pol in enumerate(pols_x2):
                    if step == 1:
                        txts.append('$X_{' + str(i+1) + ',1}=' + ','.join(pol) + '$')
                    else:
                        irl = '({})'.format(','.join(['X_{' + str(i) + ',' + str(j+1) + '}' for j in range(step)]))
                        irr = '({})'.format(','.join(pol))
                        txts.append('${}={}$'.format(irl, irr))

                # 生成输入关系表达式
                txts.append('\n**output:**')  # 输入关系
                pols_x = {'Y_{0,1}': '{}({})'.format(MR['name'], ','.join(MR['item_X'][1:]))}  # 提前在项中放入X_0
                for i, pol in enumerate(pols_x2):
                    pols_x['Y_{' + str(i + 1) + ',1}'] = '{}({})'.format(MR['name'], ','.join(pol))  # 生成Y_{i,j}的表达式

                # 得到输入关系：
                # txts.append('**IR:**')
                for p in pols_x:
                    txts.append('${}={}$'.format(p, pols_x[p]))
                txts.append('')

                # 2. 得到MR关系：
                pols_y = mat_to_polynomial(MR['OR'], MR['item_Y'])
                mrs = ';'.join(pols_y)
                mrs = mrs.split(';')
                for i, mr in enumerate(mrs):
                    txts.append('**MR{}:** ${}=0$;'.format(i + 1, mr))

            txts = [txt + '\n' for txt in txts]
            txts.append('\n')
            fw.writelines(txts)


def npz_to_json(npz_root, json_path):
    path = npz_root     # 将path指向需要读取的文件夹

    # 读取文件夹中的文件
    npz_list = os.listdir(path)
    map_MR = {}  # 存放蜕变关系字典，key为序号

    # 依次处理每一个npz文件
    for name_npz in npz_list:
        if name_npz.split('.')[-1] == 'npz':
            par = name_npz.split('_')[0:6]  # 获得参数
            par = list(map(int, par))  # 将参数转为 int
            if par[0] not in map_MR.keys():  # 第一次遇到id为par[0]的函数，将map_MR[par[0]]初始化为一个列表
                map_MR[par[0]] = []

            # 获得par[0]的函数名字，如果找不到名字，则暂时命名为 str(par[0])
            if par[0] in settings.map_index_func.keys():
                fun_name = settings.map_index_func[par[0]]
            else:
                fun_name = str(par[0])

            print(str(fun_name) + '_' + name_npz + '-------')
            t_dict = {  # 临时字典
                'name': fun_name,  # 函数名
                'num_involved_inputs': par[1],  # 当前关系的元数
                'is_equal': (par[2] == 1 and par[3] == 1),  # 是否为对等关系
                'input_degrees': par[4],  # 输入模式的阶数
                'output_degrees': par[5]  # 输出模式的阶数
            }

            if t_dict['is_equal']:  # 暂时只考虑对等关系
                # if True:
                MRs = load_phase3_results(path + name_npz)  # 读取MR关系
                if MRs is None:
                    return None
                for k, v in MRs.items():
                    # 处理列名：
                    print(len(MRs.items()))
                    latex_x = column_to_latex(v[0].columns, 'X')
                    latex_y = column_to_latex(v[1].columns, 'Y')
                    print(latex_x)
                    print(latex_y)

                    # 保存关系
                    t_dict['item_X'] = latex_x
                    t_dict['item_Y'] = latex_y
                    t_dict['IR'] = v[0].values.tolist()
                    t_dict['OR'] = v[1].values.tolist()
                map_MR[par[0]].append(t_dict)
    with open(json_path, "w") as f:
        f.write(json.dumps(map_MR, ensure_ascii=False, indent=4, separators=(',', ':')))
    return 1


if __name__ == '__main__':
    # 此脚本用于将npz文件读取成json格式和markdown格式
    json_name = 'np'    # 将json文件保存成np.json
    markdown_name = 'npmd'      # 将markdown文件保存成npmd.md
    npz_root = 'Associated_research_paper/PaperResults/NumPy/phase3/'  # npz文件目录，请最后一定要以/结尾。
    npz_to_json(npz_root, '{}.json'.format(json_name))                 # 将npz文件转换为json文件
    json_to_md('{}.json'.format(json_name), '{}.md'.format(markdown_name))      # 将json文件转化为md文件

