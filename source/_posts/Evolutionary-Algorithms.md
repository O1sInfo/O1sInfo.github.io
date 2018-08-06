---
title: Evolutionary Algorithms
date: 2018-08-05 17:12:46
tags: 进化算法
categories: 进化计算
---

>It is not strongest of the species that survives, nor the most intelligent that survives. It is the one that is the most adaptable to change. -- Charles Darwin

## Motivation of EAs

1. What can EAs do for us?
	- Optimization
	- Help people understand the evolution in nature.

2. What is optimizatin?
	- The process of searching for the optimal solution from a set of candidates to the problem of interest based on certrain **performance criteria**

3. Produce maximum yields given litmited resources.

## Key Concepts

- Population-Based Stochastic Optimization Methods
- Inherently Parallel
- A Good Example of Bionics in Engineering
- Survival of the Fittest
- Chromosome, Crossover, Mutation
- Metaheuristics
- Bio-/Nature Inspired Computing

## The Big Picture

![](/images/eas.png)

## EA Family

- GA: Genetic Algorithm
- GP: Genetic Programming
- ES: Evolution Strategies
- EP: Evolution Programming
- EDA: Estimation of Distribution Algorithm
- PSO: Particle Swarm Optimization
- ACO: Ant Colony Optimization
- DE: Differential Evolution

## Optimization Problem Set

- Portfolio Optimization
- Travelling Salesman Problem
- Knapsack Problem
- Machine Learing Problems

![](/images/local_optima.png)

Many interesting optimization problems are not trivial.The optimal solution cannot always be found in polynomial time.

## Solution: Parallel Search

- Conduct searching in different areas simultaneously.
	- Population Based
	- Avoid unfortunate starting positions.
- Employ heuristic methods to effectively explore the space.
	- Focus on promising areas.
	- Also keep an eye on other regions.
	- More than random restart strategies.

## Publications

Top Journals:
- IEEE Transactions On Evolutionary Computation.
- Evolutionary Compution Journal

Major Conference:
- IEEE Congress On Evolution Computation(CEC)
- Genetic and Evolution Computation Conference(GECCO)
- Parallel Problem Solving from Nature(PPSN)

Game:
	- Blondie24: Playing at the Edge of AI

Book:
	- How to Solve It: Modern Heuristics
