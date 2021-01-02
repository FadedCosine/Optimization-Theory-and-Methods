# 最优化算法实现

## 文件结构
该项目为最优化理论与算法的课程项目，项目文件结构如下：

```
├── Large_Scale_Methods #有限内存的拟牛顿方法
|  ├── L_BFGS.py #包括有限内存BFGS和其压缩形式的代码实现
|  └── L_SR1.py #包括有限内存SR1的压缩形式的代码实现
├── Line_Search #包含了第一次上机作业中的线搜索程序
|  ├── exact_line_search.py # 包含精确线搜索代码：进退法和0.618法的实现
|  ├── inexact_line_search.py # 非精确线搜索的代码实现，通过参数选择["Armijo Goldstein", "Wolfe Powell", "Strong Wolfe Powell"]不同的准则
|  └── GLL.py # GLL准则的代码实现
├── Newton_Methods
|  ├── fletcher_freeman.py # Fletcher Freeman方法的代码实现
|  ├── newton_method.py # 包括了阻尼牛顿法、GM稳定牛顿法的代码实现
|  └── inexact_newton_method.py #非精确牛顿法和非精确牛顿回溯法的代码实现
├── Trust_Region #包括信赖域型算法主框架，以及对信赖域子问题求解的不同算法
|  ├── hebden.py #包括Hebden方法求解信赖域子问题的代码实现
|  ├── sorensen.py #包括More-Sorensen方法求解信赖域子问题的代码实现
|  ├── trust_region_main.py #包括信赖域型算法主框架的实现，以及在测试函数上的测试代码
|  └── two_subspace_min.py #包括二维子空间极小化方法求解信赖域子问题的代码实现
├── functions.py #包括了所有的测试函数
├── newton_methods_main.py # 对牛顿型方法进行数值实验的主程序
├── README.md
└── utils.py #包括修正Cholesky分解和BP分解、以及验证矩阵是否正定的函数等工具函数的代码实现
```

### 运行步骤

对于牛顿型方法newton_methods_main.py文件中包括了所有数值实验的代码，可以直接运行newton_methods_main.py，
参数 --m 指定参数函数的维度（仅对Extended Powell singular function和Trigonometric function有效），--test_fucntion 指定测试函数（从["Wood", "EPS", "Trig"]中选择）

如：
```python
python newton_methods_main.py \
--m 20 \
--test_fucntion EPS
```

对于非精确牛顿和拟牛顿型的方法由于数值实验参数十分复杂没有整合成一个主程序，对于：

1. 非精确牛顿法和非精确牛顿回溯法(Newton_Methods/inexact_newton_method.py)，其中用参数控制不同eta的选择和安全保护的开关；
2. 有限内存BFGS和其压缩形式(Large_Scale_Methods/L_BFGS.py)；
2. 有限内存SR1的压缩形式(Large_Scale_Methods/L_SR1.py)；

这些程序的数值实验均在对应的文件中，注释或者解开对应的注释、修改对应的参数即可运行对应函数的数值实验。

对于信赖域型的方法，Trust_Region/trust_region_main.py文件中包括了所有数值实验的代码，(需要先声明python的工作路径，通过该命令： export PYTHONPATH="${PYTHONPATH}:./" )可以在当前目录直接运行Trust_Region/main.py，

如：
```python
python Trust_Region/trust_region_main.py \
--test_fucntion Trig \
--m 20
```