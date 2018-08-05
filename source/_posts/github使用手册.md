---
title: github使用手册
date: 2018-08-05 10:45:20
tags: git
categories: 程序员实用工具
---
## git clone

### clone地址https和SSH的区别

前者可以随意克隆github上的项目，而不管是谁的；而后者则是你必须是你要克隆的项目的拥有者或管理员，且需要先添加 SSH key ，否则无法克隆。

https url 在push的时候是需要验证用户名和密码的；而 SSH 在push的时候，是不需要输入用户名的，如果配置SSH key的时候设置了密码，则需要输入密码的，否则直接是不需要输入密码的。

### 在github上添加ssh key的方法

1. 	首先需要检查你电脑是否已经有 SSH key 

`cd ~/.ssh/ | ls` 检查是否有文件id_rsa.pub, 若存在则跳过第二步

2. 创建一个ssh key

`ssh-keygen -t rsa -C "your_email@example.com"` 使用默认设置，可设置密码用于push操作。完成后将得到两个文件，放在./ssh目录下，分别为id_rsa和id_rsa.pub

3. 添加ssh key到github

拷贝id_rsa.pub文件的内容，复制到github账户的sshkey设置页面处。

4. 测试ssh key

`ssh -T git@github.com`

### clone指定分支

`git clone -b <分支名> <address.git>`

## 添加新的分支

1. 先将仓库克隆到本地
2. `git branch`查看分支。`git branch <分支名>` 新建分支
3. `git checkout <分支名>` 切换到新分支
4. `git push -u origin <分支名>` 同步分支到github
