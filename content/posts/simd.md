---
title: SIMD演算のざっくり解説。
date: 2022-01-28
category: [Programming, Research-ish]
tags: [C++]
image: "/image/blog-pic.jpg"
draft: true
---

この記事は，SIMD演算について，使い方も含め，ざっくり解説してみようものです．

SIMD演算とは **S**ingle **I**nstruction / **M**ultiple **D**ata のことです．
名の通り，**一つの命令で，複数の値を操作できる命令**のことです．
DeepLearning 等の発展により，GPUによる並列計算が注目を集めています．
GPUとは，**G**raphical **P**rocessing **U**nit の略で，ゲームなどの美しいグラフィックを効率よく描写したいよね，そんな気持ちで開発されていたものです．
一般的に，行列の積などを計算する際に部分積の箇所を

