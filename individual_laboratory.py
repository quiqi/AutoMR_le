"""
author: QUQI
tel: 13142338890
email: 1292396438@qq.com
"""
import Phase2_Filter
import Phase3_RemoveRedundancy
from load_resulrs import *
import time
import csv


if __name__ == '__main__':
    # 此脚本用于实现多次独立重复实验，可以生成npz格式的实验数据和一张表格，表格记录每个函数在每个参数下在每次独立运行时花费的时间
    repet = 5       # 重复实验次数
    pars = ["2_1_1_1_1"]    # 参数设置
    fun_ids = [21]          # 待实验函数
    save_root = 'output/{}/'.format(time.strftime("%Y_%m_%d_%Hh%Mm%Ss", time.localtime()))  # 保存地址，默认保存到output下的 “当下时间” 文件夹
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    for fun_id in fun_ids:  # 每次跑一个函数
        with open(save_root + 'times_{}.csv'.format(fun_id), 'a', newline='', encoding='utf-8') as f:
            cw = csv.writer(f)
            for par in pars:    # 每次跑一个参数
                settings.func_indices = [fun_id]
                settings.func_indices = [par]
                for i in range(repet):      # 每种情况运行repet次
                    save_root_times = save_root + '{}/{}/{}/'.format(fun_id, par, i)
                    if not os.path.isdir(save_root_times):
                        os.makedirs(save_root_times)
                    settings.output_path = save_root_times
                    s = time.time()
                    print('{}-{}-{}起始时间：{}s'.format(i, settings.map_index_func[fun_id], par, s))
                    print('start phase 1: searching ...')
                    Phase1_PSOSearch.run_phase1()
                    print('start phase 2: filtering ...')
                    Phase2_Filter.run_phase2()
                    print('start phase 3: redundancy removing ...')
                    Phase3_RemoveRedundancy.run_phase3()
                    e = time.time()
                    print('{}-{}-{}花费时间：{}s'.format(i, settings.map_index_func[fun_id], par, e - s))
                    cw.writerow([settings.map_index_func[fun_id], i, par, e - s])