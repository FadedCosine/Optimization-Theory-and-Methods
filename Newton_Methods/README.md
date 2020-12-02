# 最优化第一次上机作业

## 文件结果
该项目上传的文件结构如下：

```
├── logs # 包含最终数值实验时打印出的log
├── results # 包含最终数值实验的结果
├── exact_line_search.py # 包含精确线搜索代码：进退法和0.618法的实现
├── fletcher_freeman.py # Fletcher Freeman方法的代码
├── functions.py # 包含三个测试函数及其对应的一阶导数和Hesse矩阵的代码实现
├── GLL.py # GLL准则的代码实现
├── inexact_line_search.py # 非精确线搜索的代码实现，通过参数选择["Armijo Goldstein", "Wolfe Powell", "Strong Wolfe Powell"]不同的准则
├── main.py # 数值实验的主程序
├── newton_method.py # 包括了阻尼牛顿法、GM稳定牛顿法的实现
├── README.md
└── utils.py #包括修正Cholesky分解和BP分解的代码实现
```

### 运行步骤
main.py文件中包括了所有数值实验的代码，可以总结运行main.py，
参数 --m 指定参数函数的维度（仅对Extended Powell singular function和Trigonometric function有效），--test_fucntion 指定测试函数（从["Wood", "EPS", "Trig"]中选择）

如：
```python
python main.py \
--m 20 \
--test_fucntion EPS
```