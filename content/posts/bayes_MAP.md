---
title: ベイズから眺めるMAP推定は豊洲の高層ビルから眺める東京湾よりも美しいのか。
date: 2022-01-21
tags: [Bayesian Inference, Machine Learning, Mathematical Optimization, Regression]
description: "本記事では，ベイズの枠組みでLASSO回帰やRidge回帰に代表される，正則化付き最小二乗法を眺めてみます．"
math: true
image: "/image/Bayes_MAP_head.png"
---

本記事では，ベイズの枠組みでLASSO回帰やRidge回帰に代表される，正則化付き最小二乗法を眺めてみます．
正則化は機械学習を学ぶと早い段階で出てくる，過学習を抑制するためのテクニックですよね．
この正則化も実はベイズの枠組みで見ると，MAP（事後確率最大化：Maximum a posteriori）推定に相当します．
「当たり前じゃん」と思った方には不毛な記事になりますが，是非最後まで読んでいただけると幸いです．

### 1. ベイズ推論（Bayesian Inference）

ベイズ推論については渡辺先生の[こちら]()が詳しいです．
与えられたデータ $x^n=(x_1,\cdots,x_n),~ x_i\in\mathbb R^N$ は真の確率分布$q(x)$にi.i.dに従う確率変数 $X_i\in\mathbb R^N,~i=1,\cdots,n$ の実現値と考えます．
つまり，データの同時分布は
$$
    q(x^n)=\prod_{i=1}^nq(x_i) =q(x_1)q(x_2)\cdots q(x_n),
$$
となります．
ここで，真の分布 $q(x)$ は知ることのできないものとして考えます．
ベイズ推論とは，<mark>真の分布 $q(x)$ を与えられたデータ $x^n$ から推測すること</mark>を指します．

それでは，具合的にどうやって推測を行うのでしょうか？
ここで，人間はパラメータ $w\in W\subset \mathbb R^d$ によって形状が決定する， $x\in\mathbb R^N$ 上の条件付き確率分布 $p(x|w)$ (**確率モデル**) と $w\in W$ 上の確率分布 $\varphi(w)$ (**事前分布**) を用意します．
渡辺先生の本では，逆温度 $\beta$ での事後確率が登場しますが，本記事では，データ $x^n$ が与えられたときの，パラメータ $w$ の事後確率 $p(w|X^n)$ を一般的なベイズ統計の教科書に合わせて以下で定義します．

$$
    p(w|X^n) = \frac{1}{Z}\varphi(w)\prod_{i=1}^Np(X_i|w)
$$

ここで， $Z$ は規格化定数

$$
    Z = \int_W \varphi(w)\prod_{i=1}^Np(X_i|w)\text{d}w = p(X^n),
$$

であり，**周辺尤度**と呼ばれます．


### 工事中。