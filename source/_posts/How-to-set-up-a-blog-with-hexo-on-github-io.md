---
title: How to set up a blog with hexo on github.io
date: 2018-07-18 18:17:31
tags: hexo
categories: web
---
### Install Git (https://git-scm.com/)

* If you have a git, you can check it by `git -v`

### Install node.js envornment 

* Download from [https://nodejs.org/en/](https://nodejs.org/en/)
* Install as the default(make sure the envorment path is collected)
* When you finsh, you can check it by `node -v`

###  Make a new repository named "<your_username>.github.io"

### Install Hexo

* `npm install hexo-cli -g`
* `hexo init blog`(that folder you wanted to store your webpage)
* `cd blog`
* `npm install`
* `hexo server`

### Connect hexo with github

* `cd blog`
* `git config --global user.name "<your_username>"`
* `git config --global user.email "<your_email>"`

Check if you have a ssh keygen. If not you can do as following

* `cd ~/.ssh`
* generate key: `ssh-kengen -t rsa -C "<your_email>"` (choice default setting)
* add key to ssh-agent: `eval "$(ssh-agent -s)"`
* `ssh-add ~/.ssh/id_rsa`

### Sign in the github, in settings, add a new ssh key, copy the `id_rsa.pub` to key options. check if it's ok by `ssh -T git@github.com`. If you get a hi message, it is ok.

### In your blog folder, edit the _config.yml file like this.

```
(in the end)
deploy:
    type: git
    repository: git@github.com:<your_username>/<your_username>.github.io.git
    branch: master
```

### Before deploy the blog website, you should install a plug

* `cd blog`
* `npm install hexo-deployer-git --save`

### Run it online.

* `hexo clean`
* `hexo g`
* `hexo d`

