---
title: Genetic Algorithms
date: 2018-08-05 22:14:07
tags: 遗传算法
categories: 进化计算
---

## Biology Background

- Gene: A working subunit of DNA
- Gene Trait(性状基因): For example colour of eyes
- Allele(等位基因): Possible settings for a trait
- Genotype(基因型): The actual genes carried by an individual
- Phenotype(性状型): The physical characteristics into which genes are translated

## Genetic Algorithms

By John Holland: "Adaptation in Natural and Artificial Systems"

### Inspired by and (loosely) based on Darwin's Theory

- Chromosome(染色体)
- Crossover(交叉)
- Mutation(变异)
- Selection(Survival of the Fittest)

### Basic Ideas

- Each solution to the problem is represented as a chromosome.
- The initial solutions may be randomly generated.
- Solution are evolved during generations.
- Improved gradually based on the principle of natural evolution.

### Basic Components

#### Representation

- How to encode the parameters of the problem?
- Binary Problems
- Continuous Problems

1. Individual(Chromosome)

A vector that represents a specific solution to the problem.Each element on the vector corresponds to a certain variable/parameter.

2. Population

A set of individuals, GAs maintain and evolve a population of individuals, Parallel Search to get Global Optimization.

3. Offspring

New individuals generated via genetic operators. Hopefully contain better solutions.

4. Encoding

Binary vs. Gray. How to encode TSP problems?

#### Genetic Operators

1. Crossover:
	Exchange genetic materials between two chromosomes.
	- One Point Crossover
	- Two Point Crossover
	- Uniform Crossover
 
2. Mutation:
	Randomly modify gene values at selected locations.Mutation is mainly used to maintain the genetiv divesity.Loss of genetic diversity will result in Permature Convergence.

#### Selection Strategy

- Which chromosomes should be involved in reproduction?
- Which offspring should be able to survive?

1. Roulette Wheel Selection: 根据适应度的高低按比例选择
2. Rank Selection： 根据排名按固定比例选择
3. Tournament Selection: 两两及以上互相竞争
4. Elitism: 精英保留直接拷贝到下一代
5. Offspring Selection: 子代直接进入下一代还是与父代一起竞争

### Selection vs. Crossover vs. Mutation

- Selection:
	- Bias the search effort towards promising individuals.
	- Loss of genetic diversity

- Cossover:
	- Create better individuals by combining genes from good individuals
	- Building Block Hypothesis
	- Major search power of GAs
	- No effect on genetic diversity

- Mutation:
	- Increase genetic diversity
	- Force the algorithm to search areas other than the current focus.

**It is a trade off about Exploration vs. Exploitation**

## GA Framework

1. Intialization: Generate a random population P of M individuals
2. Evaluation: Evaluate the fitness f(x) of each individual
3. Repeat until the stopping criteria are met:
	1. Reproduction: Repeat the following steps until all offspring are generated
		1. Paraent Selection: Select two parents from P
		2. Crossover: Apply crossover on the parents with probability P_c
		3. Mutation: Apply mutation on offspring with probability P_m
		4. Evaluation: Evaluate the newly generated offspring
	2. Offspring Selection: Create a new population from oddspring and P
	3. Output: Return the best individual found

## Parameters

- Population Size:
	Too big: Slow convergence rate. Too small: Premature convergence

- Crossover Rate:
	Recommended value: 0.8

- Mutation Rate:
	Recommeded value: 1/L. Too big: Disrupt the evolution process. Too small: Not enough to maintain diversity.

- Selection Strategy:
	Tournament Selection. Truncation Selection(Select top T individuals). Need to be careful about the selection pressure.
