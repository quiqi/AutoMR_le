# AutoMR 修改版
> 原AutoMR的git地址：https://github.com/bolzzzz/AutoMR

这是修改版的AutoMR，相较于原版而言，修改了：
1. bug: setting.py 中 map_index_func的键值错误（有两个值为15的键值）
2. bug: setting.py 中 更详细的配置了每个函数的定义域，见get_input_range函数的修改
3. bug: Phase3_RemoveRedundancy.py 中，427行缩进错误

添加了：
1. 脚本 individual_laboratory.py，对每个函数的每个参数进行独立的实验。
2. 脚本 get_input_and_output.py，用于生成随机的输入输出数据。
3. 脚本 npz2md.py，用于将npz文件提取成json文件和markdown文件。
4. 数据 Associated_research_paper/OurResults。
5. 文档 AutoMR使用教程，更详细的介绍了AutoMR的使用方法。

> 有任何问题欢迎联系:
> 1292396438@qq.com