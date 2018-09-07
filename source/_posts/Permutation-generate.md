---
title: Permutation generate
date: 2018-09-04 09:49:58
tags: Permutation
categories: 算法导论
---

### Permutation

In mathematics, the notion of permutation relates to the act of arranging all the members of a set into some sequence or order, or if the set is already ordered, rearranging (reordering) its elements, a process called permuting. These differ from combinations, which are selections of some members of a set where order is disregarded.

### Algorithms to generate permutations

#### Random generation of permutations

The **Fisher–Yates shuffle** is an algorithm for generating a random permutation of a finite sequence—in plain terms, the algorithm shuffles the sequence. The algorithm effectively puts all the elements into a hat; it continually determines the next element by randomly drawing an element from the hat until no elements remain. The algorithm produces an unbiased permutation: every permutation is equally likely.

The basic method given for generating a random permutation of the numbers 1 through N goes as follows:

```python
# 1. Write down the numbers from 1 through N.
# 2. Pick a random number k between one and the number of unstruck numbers remaining (inclusive).
# 3. Counting from the low end, strike out the kth number not yet struck out, and write it down at the end of a separate list.
# 4. Repeat from step 2 until all the numbers have been struck out.
# 5. The sequence of numbers written down in step 3 is now a random permutation of the original numbers.

scatch = [1, 2, 3, 4, 5]
result = []
for i in range(len(scatch)):
    roll = random.randint(1, len(scatch))
    struck = scatch.pop(roll-1)
    result.append(struck)
```

**The modern version of the algorithm** is efficient: it takes time proportional to the number of items being shuffled and shuffles them in place. The modern version of the Fisher–Yates shuffle, designed for computer use, was introduced by Richard Durstenfeld in 1964[2] and popularized by Donald E. Knuth in **The Art of Computer Programming** as "Algorithm P (Shuffling)".

```python
# To shuffle an array a of n elements (indices 0..n-1):
for i from n−1 downto 1 do
     j ← random integer such that 0 ≤ j ≤ i
     exchange a[j] and a[i]

# An equivalent version which shuffles the array in the opposite direction (from lowest index to highest) is:

for i from 0 to n−2 do
     j ← random integer such that i ≤ j < n
     exchange a[i] and a[j]
```

**Sattolo's algorithm**

A very similar algorithm was published in 1986 by Sandra Sattolo for generating uniformly distributed cycles of (maximal) length n.[6][7] The only difference between Durstenfeld's and Sattolo's algorithms is that in the latter, in step 2 above, the random number j is chosen from the range between 1 and i−1 (rather than between 1 and i) inclusive. This simple change modifies the algorithm so that the resulting permutation always consists of a single cycle.

```python
from random import randrange

def sattoloCycle(items):
    i = len(items)
    while i > 1:
        i = i - 1
        j = randrange(i)  # 0 <= j <= i-1
        items[j], items[i] = items[i], items[j]
```

#### Generation in lexicographic order

The following algorithm generates the next permutation lexicographically after a given permutation. It changes the given permutation in-place.

```python
# 1. Find the largest index k such that a[k] < a[k + 1]. If no such index exists, the permutation is the last permutation.
# 2. Find the largest index l greater than k such that a[k] < a[l].
# 3. Swap the value of a[k] with that of a[l].
# 4. Reverse the sequence from a[k + 1] up to and including the final element a[n].

k = -1
l = -1
for i in reversed(range(len(a) - 1)):
    if a[i] < a[i + 1]:
        k = i
        break
if k == -1:
    return
for j in reversed(range(len(a))):
    if a[j] > a[k]:
        l = j
        break
a[k], a[l] = a[l], a[k]
a= a[:k + 1] + list(reversed(a[k + 1:]))
```

#### Generation with minimal changes

**Heap's algorithm** generates all possible permutations of n objects. It was first proposed by B. R. Heap in 1963.[1] The algorithm minimizes movement: it generates each permutation from the previous one by interchanging a single pair of elements; the other n−2 elements are not disturbed.

Suppose we have a permutation containing n different elements. Heap found a systematic method for choosing at each step a pair of elements to switch, in order to produce every possible permutation of these elements exactly once. Let us describe Heap's method in a recursive way. First we set a counter i to 0. Now we perform the following steps repeatedly until i is equal to n. We use the algorithm to generate the (n−1)! permutations of the first n−1 elements, adjoining the last element to each of these. This generates all of the permutations that end with the last element. Then if n is odd, we switch the first element and the last one, while if n is even we can switch the ith element and the last one (there is no difference between n even and odd in the first iteration). We add one to the counter i and repeat. In each iteration, the algorithm will produce all of the permutations that end with the element that has just been moved to the "last" position. The following pseudocode outputs all permutations of a data array of length n.

```python
def generate(n, A):
    if n == 1:
          return A
    else:
        for i in range(n-1):
            generate(n - 1, A)
            if n % 2 == 0:
                A[n-1], A[i] = A[i], A[n-1]
            else:
                A[n-1], A[0] = A[0], A[n-1]
        generate(n - 1, A)

# a non-recursive format
def generate(n, A):
    c = []
    for i in range(n):
        c[i] = 0
    print(A)
    i = 0
    while i < n:
        if  c[i] < i:
            if i % 2 == 0:
                A[i], A[0] = A[0], A[i]
            else:
                A[i], A[c[i]] = A[c[i]], A[i]
            print(A)
            c[i] = c[i] + 1
            i = 0
        else:
            c[i] = 0
            i = i + 1
```
