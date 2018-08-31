---
title: Requests
date: 2018-08-14 08:15:18
tags: python
categories: python包和模块
---

## [Requests](cn.python-requests.org)

Requests is an elegant and simple HTTP library for python, built for human beings.

## Beloved Features

* Keep-Alive & Connection Pooling
* International Domains and URLs
* Session with Cookie Persistence
* Browser-style SSL Verification
* Automatic Content Decoding
* Basic/Digest Authentication
* Elegant Key/Value Cookie
* Automatic Decompression
* Unicode Response Bodies
* HTTP(S) Proxy Support
* Multipart File Uploads
* Streaming Downloads
* Connection Timeouts
* Chunked Requests
* `.netrc` Support

## 快速上手

### 发送请求

```python
import requests
r = requests.get(url)
```
`r`为Response对象，requests的方法还有`put, delete, head, options`

### 传递URL参数

如url为`http://httpbin.org/get?key2=value2&key1=value1`
```python
payload = {'key1':'value1', 'key2':'value2'}
r = requests.get("http://httpbin.org/get", params=payload)
```

### 响应内容

`r.text` 响应文本
`r.encoding` 编码格式

### 二进制响应内容

`r.content` 以字节的方式访问响应体

```python
from PIL import Image
from io import BytesIO
i = Image.open(BytesIO(r.content))
```

### JSON响应内容

`r.josn()`

### 原始响应内容

```python
r = requests.get(url, stream=True)
r.raw
r.raw.read(10)

with open(filename, 'wb') as fd:
    for chunk in r.iter_content(chunk_size):
        fd.write(chunk)
```

### 定制请求头

`headers = {'user-agent': 'my_-app/0.0.1'}`
`r = requests.get(url, headers=headers)`

### 更复杂的POST请求

```python
payload = {'key1':'value1', 'key2':'value2'}
r = requests.post("http://httpbin.org/get", data=payload)

payload = (('key1', 'value1'), ('key2', 'value2'))
r = requests.post("http://httpbin.org/get", data=payload)

r = requests.post(url, json=payload)
```

### POST一个多部分编码的文件

```python
files = {'file': open('report.xls', 'rb')}
r = requests.post(url, files=files)

# 你还可以显式地设置文件名，文件类型和请求头
files = {'file': ('report.xls', open('report.xls', 'rb'), 'application/vnd.ms-excel', {'Expires': '0'})}
```

### 响应状态码

`r.status_code` 检查响应状态码
`r.status_code == requests.codes.ok` 内置的状态码查询对象
`r.raise_for_status()` 抛出异常

### 响应头

`r.headers`
```json
{
    'content-encoding': 'gzip',
    'transfer-encoding': 'chunked',
    'connection': 'close',
    'server': 'nginx/1.0.4',
    'x-runtime': '148ms',
    'etag': '"e1ca502697e5c9317743dc078f67693f"',
    'content-type': 'application/json'
}
```

### Cookie

`r.cookies`
```python
cookies = dict(cookies_are='working')
r = requests.get(url, cookies=cookies)
```

Cookie 的返回对象为 RequestsCookieJar，它的行为和字典类似，但接口更为完整，适合跨域名跨路径使用。你还可以把 Cookie Jar 传到 Requests 中：

```python
jar = requests.cookies.RequestsCookieJar()
jar.set('tasty_cookie', 'yum', domain='httpbin.org', path='/cookies')
r = requests.get(url, cookies=jar)
```

### 重定向和请求历史

Response.history 是一个 Response 对象的列表，为了完成请求而创建了这些对象。这个对象列表按照从最老到最近的请求进行排序。

```python
r = requests.get(url, allow_redirects=False)
r.history
```

### 超时

`r = requests.get(url, timeouts=0.01)`

### 错误与异常

遇到网络问题（如：DNS 查询失败、拒绝连接等）时，Requests 会抛出一个 ConnectionError 异常

如果 HTTP 请求返回了不成功的状态码， Response.raise_for_status() 会抛出一个 HTTPError 异常

若请求超时，则抛出一个 Timeout 异常。

若请求超过了设定的最大重定向次数，则会抛出一个 TooManyRedirects 异常。

所有Requests显式抛出的异常都继承自 requests.exceptions.RequestException 。
