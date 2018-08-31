---
title: matplotlib
date: 2018-08-16 07:51:58
tags: python
categories: python包和模块
---

## 快速绘图

### 使用pyplot模块绘图

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8,4))
plt.plot(x, y, label="$sin(x)$", color="red", linewidth=2)
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("pyplot first example")
plt.ylim(-1.2, 1.2)
plt.legend()
plt.show()
```

保存图片`plt.savefig('test.png', dpi=120)`的像素值由参数`matplotlib.rcParams["savefig.dpi"]`决定，默认为100.
保存对象不一定是文件，还可是和文件对象有相同调用接口的对象.

```python
from StringIO import StringIO
buf = StringIO()
plt.savefig(buf, fmt='png')
buf.getvalue()[:20]
```

### 以面向对象方式绘图

```python
fig = plt.gcf()  # get current figure
axes = plt.gca()  # get current axes
```

在pyplot模块中，许多函数都是对当前的Figure和Axes对象进行处理.

### 配置属性

使用matplotlib绘制的图表的每个组成部分都和一个对象对应，可以通过调用这些对象的属性设置方法`set_*()`或pyplot模块的属性设置函数`setp()`来设它们的属性值.

```
x = np.arange(0, 5, 0.1)
line = plt.plot(x, x*x)[0]
line.set_antialiased(False)

lines = plt.plot(x, np.sin(x), x, np.cos(x))
plt.setp(lines, color="r", linewidth=2.0)
```

同样可以调用Line2D对象的`get_*()`或`plt.getp()`来获取对象的属性值.

```python
line.get_linewidth()

# getp()只能对一个对象操作
plt.getp(lines[0], "color")
plt.getp(lines[1])  # 输出全部属性

f = plt.gcf()
plt.getp(f)

allines = plt.getp(plt.gca(), "lines")
allines = f.axes[0].lines
```

### 绘制多个子图

一个Figure对象可以包含多个子图Axes.

`subplot(numRows, numCols, plotNum)`

`subplot(323), subplot(3, 2, 3)`

```python
# 绘制6个子图并设置不同的背景颜色
for idx, color in enumerate("rgbyck"):
    plt.subplot(321 + idx, axisbg=color)
plt.show()
```

`plt.subplot(212)  # 占据第二整行`

```python
同时在多幅图表、多个子图中进行绘制
import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)  # 创建图表1
plt.figure(2)
ax1 = plt.subplot(211)  # 在图表2中创建子图1
ax2 = plt.subplot(212)

x = np.linspace(0, 3, 100)
for i in xrange(5):
    plt.figure(1)  # 选择图表1
    plt.plot(x, np.exp(i * x / 3)
    plt.sca(ax1)  # 选择图表2的子图1
    plt.plot(x, np.sin(x * i))
    plt.sca(ax2)  # 选择图表2的子图2
    plt.plot(x, np.cos(i * x))
plt.show()
```

### 配置文件

绘制一幅图表要对许多对象的属性进行配置。我们通常采用了默认配置，matplotlib将这些默认配置保存在一个名为“matplotlibrc”的配置文件中。

```python
matplotlib.get_configdir()  # 获取用户配置路径
matplotlib.matplotlib_fname()  # 获得目前使用的配置文件的路径
matplotlib.rc_params()  # 配置文件的读入，返回字典
matplotlib.rc("lines", marker='x', linewidth=2, color="red")  # 对配置字典进行设置
matplotlib.rcdefaults()  # 回复默认配置
``````

### 在图表中显示中文

```pythno
from matplotlib.font_manager import fontManager
# 获得所有可用的字体列表
fontManager.ttflist

# 获得字体文件的全路径和字体名
fontManager.ttflist[0].name
fontManager.ttflist[0].fname


```python
# 显示所有的中文字体
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
import os

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
plt.subplot_adjust(0, 0, 1, 1, 0, 0)
plt.xticks([])
plt.yticks([])
x, y = 0.05, 0.08
fonts = [font.name for font in fontManager.ttflist if os.path.exists(font.fname) and os.stat(font.fname).st_size>1e6]
font = set(fonts)
dy = (1.0 - y) / (len(fonts) / 4 + (len(fonts) % 4 != 0))
for font in fonts:
    t = ax.text(x, y, u"中文字体", {'fontname': font, 'fontsize': 14}, transform=ax.transAxes)
    ax.test(x, y - dy / 2, font, transform=ax.transAxes)
    x += 0.25
    if x >= 1.0:
        y += dy
        x = 0.05
plt.show()
```

```python
# 使用ttc字体文件
from matplotlib.font_Manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
t = np.linspace(0, 10, 100)
y = np.sin(t)
plt.plot(t, y)
plt.title(u"正弦波", fontproperties=font)
plt.show()
```

直接修改配置文件，设置默认字体。

`plt.rcParams["font.family"] = "SimHei"`

## Artist对象
