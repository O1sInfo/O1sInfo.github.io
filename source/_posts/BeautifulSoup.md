---
title: BeautifulSoup
date: 2018-08-15 11:40:51
tags: 网页解析
categories: python包和模块
---
## [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/index.zh.html)

You didn't write that awful page. You're just trying to get some data out of it. Beautiful Soup is here to help. Since 2004, it's been saving programmers hours or days of work on quick-turnaround screen scraping projects.

## 如何使用

```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(open('index.html'))
```

## 对象的种类

### Tag

Tag 对象与XML或HTML原生文档中的tag相同:

```python
soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')
tag = soup.b
type(tag)
# <class 'bs4.element.Tag'>
```

Tag的属性：`tag.name, tag.attrs`

tag的属性的操作方法与字典相同: `tag['class']`

### 可以遍历的字符串

字符串常被包含在tag内.Beautiful Soup用 NavigableString 类来包装tag中的字符串:`tag.string`

`unicode_string = unicode(tag.string)`

tag中包含的字符串不能编辑,但是可以被替换成其它的字符串,用 replace_with() 方法:`tag.string.replace_with("No")`

### BeautifulSoup

BeautifulSoup 对象表示的是一个文档的全部内容.大部分时候,可以把它当作 Tag 对象,它支持 遍历文档树 和 搜索文档树 中描述的大部分的方法.

### 注释及特殊字符串

Comment 对象是一个特殊类型的 NavigableString 对象:

```python
markup = "<b><!--Hey, buddy. Want to buy a used parser?--></b>"
soup = BeautifulSoup(markup)
comment = soup.b.string
type(comment)
# <class 'bs4.element.Comment'>
```

## 遍历文档树

### 子节点

一个Tag可能包含多个字符串或其它的Tag,这些都是这个Tag的子节点.

#### tag的名字

`soup.head, soup.title, soup.body.b, soup.a, soup.find_all('a')`

#### .contents和.children

tag的 .contents 属性可以将tag的子节点以列表的方式输出:

通过tag的 .children 生成器,可以对tag的子节点进行循环:

```python
for child in title_tag.children:
    print(child)
```
descendants 属性可以对所有tag的子孙节点进行递归循环:

```python
for child in title_tag.descendants:
    print(child)
```

### .string

如果tag只有一个 NavigableString 类型子节点,那么这个tag可以使用 .string 得到子节点:`tag.string`

如果一个tag仅有一个子节点,那么这个tag也可以使用 .string 方法,输出结果与当前唯一子节点的 .string 结果相同

如果tag包含了多个子节点,tag就无法确定 .string 方法应该调用哪个子节点的内容, .string 的输出结果是 None :

### .strings和stripped__strings

如果tag中包含多个字符串 ,可以使用 .strings 来循环获取, 输出的字符串中可能包含了很多空格或空行,使用 .stripped_strings 可以去除多余空白内容

```python
for string in soup.strings:
    print(repr(string))

for string in soup.stripped_strings:
    ...
```

### 父节点

#### .parent

`tag.parent, tag.string.parent`

#### .parents

```python
link = soup.a
for parent in link.parents:
    if parent is None:
        print(parent)
    else:
        print(parent.name)
```

### 兄弟节点

#### .next_sibling和.previous_sibling

实际文档中的tag的 .next_sibling 和 .previous_sibling 属性通常是字符串或空白.

#### .next_siblings和.previous_siblings

通过 .next_siblings 和 .previous_siblings 属性可以对当前节点的兄弟节点迭代输出

```python
for sibling in soup.a.next_siblings:
    print(repr(sibling))
```

### 回退和前进

```html
<html><head><title>The Dormouse's story</title></head>
<p class="title"><b>The Dormouse's story</b></p>
```

HTML解析器把这段字符串转换成一连串的事件: “打开<html>标签”,”打开一个\<head>标签”,”打开一个\<title>标签”,”添加一段字符串”,”关闭\<title>标签”,”打开<p>标签”,等等.Beautiful Soup提供了重现解析器初始化过程的方法

#### .next_element和.previous_element

.previous_element 属性刚好与 .next_element 相反,它指向当前被解析的对象的前一个解析对象, next_element 属性指向解析过程中下一个被解析的对象(字符串或tag), 

#### .next_elements和.previous_elements

通过 .next_elements 和 .previous_elements 的迭代器就可以向前或向后访问文档的解析内容,就好像文档正在被解析一样:

```python
for elementin a_tag.next_elements:
    print(repr(element))
```

## 搜索文档树

### 过滤器

#### 字符串

最简单的过滤器是字符串.在搜索方法中传入一个字符串参数,Beautiful Soup会查找与字符串完整匹配的内容,下面的例子用于查找文档中所有的\<b>标签:

`soup.find_all('b')`

#### 正则表达式

如果传入正则表达式作为参数,Beautiful Soup会通过正则表达式的 match() 来匹配内容.下面例子中找出所有以b开头的标签,这表示\<body>和\<b>标签都应该被找到:

```python
import re
for tag in soup.find_all(re.compile("^b"):
    print(tag.name)
```

#### 列表

如果传入列表参数,Beautiful Soup会将与列表中任一元素匹配的内容返回.下面代码找到文档中所有\<a>标签和\<b>标签:

`soup.find_all(['a', 'b'])`

#### True

True 可以匹配任何值,下面代码查找到所有的tag,但是不会返回字符串节点

```python
for tag in soup.find_all(True):
    print(tag.name)
```

#### 方法

如果没有合适过滤器,那么还可以定义一个方法,方法只接受一个元素参数,如果这个方法返回 True 表示当前元素匹配并且被找到,如果不是则反回 False

下面方法校验了当前元素,如果包含 class 属性却不包含 id 属性,那么将返回 True:

```python
def has_class_but_no_id(tag):
    return tag.has_attr('class') and not tag.has_attr('id')
```

下面代码找到所有被文字包含的节点内容

```python
from bs4 import NavigableString

def surrounded_by_strings(tag):
    return (isinstance(tag.next_element, NavigableString)) and isinstance(tag.previous_element, NavigableString)
```

### find_all()

`find_all(name, attrs, recursive, text, **kwargs)`

find_all() 方法搜索当前tag的所有tag子节点,并判断是否符合过滤器的条件

#### name参数

搜索 name 参数的值可以使任一类型的 过滤器 ,字符窜,正则表达式,列表,方法或是 True .

name 参数可以查找所有名字为 name 的tag,字符串对象会被自动忽略掉. 

#### keyword参数

`soup.find_all(id='link2')`
`soup.find_all(href=re.compile('elsie'))`
`soup.find_all(id=True)`
`soup.find_all(attrs={'id':'link2'})`

#### 按CSS搜索

通过 ".class__"_ 参数搜索有指定CSS类名的tag, ".class__"_ 参数同样接受不同类型的 过滤器 ,字符串,正则表达式,方法或 True

`soup.find_all('a', class_='sister')`

#### text参数

通过 text 参数可以搜搜文档中的字符串内容.与 name 参数的可选值一样, text 参数接受 字符串 , 正则表达式 , 列表, True.

`soup.find_all(text=re.compile("Dormouse"))`

#### limit参数

find_all() 方法返回全部的搜索结构,如果文档树很大那么搜索会很慢.如果我们不需要全部结果,可以使用 limit 参数限制返回结果的数量.效果与SQL中的limit关键字类似,当搜索到的结果数量达到 limit 的限制时,就停止搜索返回结果.

`soup.find_all('a', limit=2)`

#### recursive参数

调用tag的 find_all() 方法时,Beautiful Soup会检索当前tag的所有子孙节点,如果只想搜索tag的直接子节点,可以使用参数 recursive=False.

`soup.find_all('title', recursive=False)`

### 像调用find_all()一样调用tag

find_all() 几乎是Beautiful Soup中最常用的搜索方法,所以我们定义了它的简写方法. BeautifulSoup 对象和 tag 对象可以被当作一个方法来使用,这个方法的执行结果与调用这个对象的 find_all() 方法相同,下面两行代码是等价的:

`soup.find_all('a')`
`soup('a')`

### find()

`find(name, attrs, recursive, text, **kwargs)`

```python
soup.find_all('title', limit=1)
# [<title>The Dormouse's story</title>]

soup.find('title')
# <title>The Dormouse's story</title>
```

唯一的区别是 find_all() 方法的返回结果是值包含一个元素的列表,而 find() 方法直接返回结果.

find_all() 方法没有找到目标是返回空列表, find() 方法找不到目标时,返回 None.

### find_parents()和find_parent()

find_parents() 和 find_parent() 用来搜索当前节点的父辈节点,搜索方法与普通tag的搜索方法相同

### find_next_siblings()和find_next_sibling()

这2个方法通过 .next_siblings 属性对当tag的所有后面解析的兄弟tag节点进行迭代, find_next_siblings() 方法返回所有符合条件的后面的兄弟节点, find_next_sibling() 只返回符合条件的后面的第一个tag节点.

### find_previous_siblings和find_previous_sibling()

find_previous_siblings() 方法返回所有符合条件的前面的兄弟节点, find_previous_sibling() 方法返回第一个符合条件的前面的兄弟节点:

### find_all_next()和find_next()

这2个方法通过 .next_elements 属性对当前tag的之后的tag和字符串进行迭代, find_all_next() 方法返回所有符合条件的节点, find_next() 方法返回第一个符合条件的节点:

### find_all_previous()和find_previous()

这2个方法通过 .previous_elements 属性对当前节点前面的tag和字符串进行迭代, find_all_previous() 方法返回所有符合条件的节点, find_previous() 方法返回第一个符合条件的节点.

### CSS选择器

通过tag标签逐层查找 `soup.select("body a")`

找到某个tag标签的直接子标签 `soup.select("p > #link1")`

找到兄弟节点标签 `soup.select("#link1 + .sister")`

通过CSS类名查找 `soup.select(".sister"), soup.select("[class~=sister]")`

通过tag的id查找 `soup.select("#link1")`

通过是否存在某个属性查找 `soup.slect('a[href]')`

通过属性的值来查找 `soup.select('a[href^="https:"]', [href$='title'], [href*=".com/"]`

## 修改文档树

## 输出

### 格式化输出

prettify() 方法将Beautiful Soup的文档树格式化后以Unicode编码输出,每个XML/HTML标签都独占一行

```python
markup = '<a href="http://example.com/">I linked to <i>example.com</i></a>'
soup = BeautifulSoup(markup)
soup.prettify()
# '<html>\n <head>\n </head>\n <body>\n  <a href="http://example.com/">\n...'

print(soup.prettify())
# <html>
#  <head>
#  </head>
#  <body>
#   <a href="http://example.com/">
#    I linked to
#    <i>
#     example.com
#    </i>
#   </a>
#  </body>
# </html>
```

### 压缩输出

如果只想得到结果字符串,不重视格式,那么可以对一个 BeautifulSoup 对象或 Tag 对象使用Python的 unicode() 或 str() 方法:

`str(soup), Unicode(soup.a)`

### 输出格式

Beautiful Soup输出是会将HTML中的特殊字符转换成Unicode,比如“\&lquot;”:
如果将文档转换成字符串,Unicode编码会被编码成UTF-8.这样就无法正确显示HTML特殊字符了:

### get_text()

如果只想得到tag中包含的文本内容,那么可以用 get_text() 方法,这个方法获取到tag中包含的所有文本内容包括子孙tag中的内容,并将结果作为Unicode字符串返回:

可以通过参数指定tag的文本内容的分隔符, 还可以去除前后空白: `soup.get_text('|', strip=True)`

## 指定文档解析器

## 编码
