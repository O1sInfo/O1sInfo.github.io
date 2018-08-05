---
title: python内置小工具
date: 2018-08-05 07:24:47
tags: 程序员实用工具
categories: 程序员实用工具
---

## 极简文件下载（Web）服务器

### 作用

快速共享文件

### 实用方法

In python2：

`python -m SimpleHttpServer`

In python3:

`python -m http.server`

执行上述命令会在当前目录启动一个文件下载服务器，默认端口8000。**若当前目录存在一个名为`index.html`的文件，则默认会显示该文件的内容**

## 使用python解压zip压缩包

`$ python -m zipfile
Usage:
    zipfile.py -l zipfile.zip        # Show listing of a zipfile
    zipfile.py -t zipfile.zip        # Test if a zipfile is valid
    zipfile.py -e zipfile.zip target # Extract zipfile into target dir
    zipfile.py -c zipfile.zip src ... # Create zipfile from sources
`