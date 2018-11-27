# 相关文件介绍
## 目录树
目录树如下所示
```
.
├── mylearn
│   ├── __init__.py
│   ├── __main__.log
│   ├── __pycache__
│   ├── logger/
│   ├── metrics/
│   ├── model_selection/
│   ├── neural_network/
│   ├── test
│   └── tree
└── readme.md
```
### neural_network/
实现了神经网络类的文件夹。
该文件夹内定义了类`MLPClassifier`和`MLPRegressor`,使用神经网络实现分类和回归。其中这两个类各自定义在`multilayer_percepron.py`和`multilayer_perceptron_regressor.py`内

由于Python的package机制，在调用时只需要按如下方式即可。
``` python
from nerual_network import MLPClassifier
from nerual_network import MLPRegressor
```
### metrics/
该目录定义了一系列度量函数，尤其是在神经网络中会使用到的度量函数。
在文件`logistic_function.py`中定义了如下的几个函数
``` python
def ReLu(dataset):
    ...
def LeakReLu(slope, dataset):
    ...
def Sigmoid(dataset):
    ...
def Softmax(dataset):
    ...
def Id(dataset):
    ...
```
其中Id函数直接返回其输入值。
该文件中还定义了以上所有函数的导数，用于替换神经网络类的预定义的一些函数。

由于package机制，按如下方式调用即可
``` python
from metrics import ReLu
```

### logger/
该目录定义了用于log的辅助函数。使用方法如下
``` python
from logger import get_logger
mylogger = get_logger(__name__)
mylogger.debug('hello world')
```

### test/
该目录定义了一系列单元测试脚本。运行方法
``` python
# 在根目录下
python -m test.<module>
```
例如
``` python
python -m test.metrics
```
### tree/
定义了决策树的文件夹。在本次项目中没有用到。