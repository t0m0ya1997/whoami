---
title: GMMのベイズ推論
date: 2022-02-08
tags: [tag1, tag2]
description: "この記事では混合ガウス分布モデル（GMM）のベイズ推論を解説します。一般的に多くの現場ではEMアルゴリズムによる推定が用いられていますが、今回はMCMCを用いてベイズ推論してみます。"
image: "/image/blog-pic.jpg"
math: true
---

### はじめに
この記事では混合ガウス分布モデル（GMM：Gaussian Mixture Model）のベイズ推論を解説します．
[Scikit-Learn]() などのライブラリではEMアルゴリズムによる推定が用いられていますが，今回はMCMCを用いてベイズ推論してみます．

また，このモデルは局所解が存在します．
MCMCは確率的な手法で，局所解へのトラップもある程度解消されるのですが，やはりトラップされるものです．
今回は，**Parallel Tempering** としても知られている**交換モンテカルロ法**(REMC: Replica Exchange Monte Carlo)を用いて局所解にトラップされないアルゴリズムを構築します．
さらに，REMCを用いると，<mark>混合数の事後分布も計算でき，適切な混合数の推定</mark>も行うことができます．

---

### GMM の定式化

GMM の定式化をします．
手元にある $n$ 個の $d$ 次元データを $x^n = (\bm x_1,\bm x_2, \cdots, \bm x_n),\bm x_i\in\mathbb R^d~(i=1,2,\cdots,n)$ とします．
各データは，以下の$K$ 個のガウス分布の混合から発生したと仮定します．

$$
    p(\bm x|\theta=\\{\pi,\mu,\Sigma\\}) = \sum_{k=1}^K\mathcal \pi_kN(\bm x|\bm\mu,\Sigma)
    \tag{1}
$$

各混合における相対的なデータ量の比を以下の<mark>混合比</mark> $\pi$ とします．

$$
    \pi = (\pi_1,\pi_2,\cdots,\pi_K),~\sum_{k=1}^K\pi_k = 1
    \tag{2}
$$

各混合のガウス分布の<mark>平均（中心）</mark>を $\mu=(\bm\mu_1,\bm\mu_2,\cdots,\bm\mu_K)$ とします．
また，各混合のガウス分布の<mark>分散・共分散行列</mark>を $\Sigma=(\Sigma_1,\Sigma_2,\cdots,\Sigma_K)$ とします．
以上が，混合正規分布の形状を決定するパラメータになります．
これらをまとめて $\theta=\\{\pi,\mu,\Sigma\\}$ と書くことにします．

さらに，各データはi.i.dに発生したと仮定すると，確率モデルは

$$
    p(x^n|\theta) = \prod_{i=1}^np(\bm x_i|\theta)=\prod_{i=1}^n\left\\{\sum_{k=1}^K\pi_k\mathcal N(\bm x_i|\bm\mu_k,\Sigma_k)\right\\}
    \tag{3}
$$

と書くことができます．

GMMの推論を行うにあたって，**潜在変数**を用いて定式化を行うと便利です．
潜在変数とは，各データ点について，そのデータ点が「<mark>どの混合から発生しているのか</mark>」を明示的に表現する変数です．
つまり，各データ点に対して$K$ 次元の One-hot なベクトルで所属する混合を表現します．
潜在変数を $l^n$ とします．

$$
    l^n = (\bm l_1,\bm l_2,\cdots, \bm l_n),　\bm l_i\in\\{0, 1\\}^K,　||\bm l_i|| = 1,　(i=1,2,\cdots,n)
    \tag{4}
$$

潜在変数を使うと式(3)の確率モデルは

$$
    p(x^n, l^n|\theta) = \prod_{i=1}^np(\bm x_i,\bm l_i|\theta)=\prod_{i=1}^n\prod_{k=1}^K\left\\{\pi_k\mathcal N(\bm x_i|\bm\mu_k,\Sigma_k)\right\\}^{l_{i,k}}
    \tag{5}
$$

と書けます．
ここで，$l_{i,k}$ は $\bm l_i$ の $k$ 番目の要素を表します．
式(5)の確率モデルを $l^n$ について周辺化すると式(3)の確率モデルとなることもわかります．

よって式(5)の確率モデルを，今回扱う**GMMの確率モデル**とします．

また，このモデルの確率変数間の関係を示す**グラフィカルモデル**は以下です．

---

### GMMのベイズ推論

GMMのベイズ推論について述べます．
グラフィカルモデルから，全変数の同時分布は

$$
    p(x^n,l^n,\theta,K) = p(x^n,l^n|\theta)p(\theta|K)p(K)
$$

となります．
$p(x^n,l^n|\theta)$ は**尤度関数**，$p(\theta|K), p(K)$ は**事前分布**です．
ここで，パラメータ $\theta$ および，潜在変数 $l^n$ が与えられた時のモデルの**誤差関数** $E(\theta,l^n)$ を，尤度関数の対数の符号反転

$$
    E(\theta,l^n) = -\log p(x^n,l^n|\theta)
$$

と定義します．
定義から，

$$
    p(x^n,l^n|\theta) = \exp\\{-E(\theta,l^n)\\}
$$

となります．

潜在変数 $l^n$，パラメータ $\theta$ および混合数 $K$ の事後分布は

$$
    p(l^n,\theta, K|x^n) = \frac{p(x^n,l^n,\theta,K)}{p(x^n)} \propto p(x^n,l^n|\theta)p(\theta|K)p(K)
$$

と書けます．
ここで，混合数 $K$ の事後分布を考えてみます．
混合数 $K\in\\{1, 2, \cdots, K_{\text{max}}\\}$ の事前分布 $p(K)$ は $\\{1, 2, \cdots, K_{\text{max}}\\}$ 上で一様分布であるとすると以下になります．

$$
    p(K|x^n) = \iint p(l^n,\theta, K|x^n)\text{d}\theta\text{d}l^n \propto \iint p(x^n,l^n|\theta)p(\theta|K)\text{d}\theta\text{d}l^n = Z(K)
$$

ここで，$Z(K)$ は**周辺尤度**と呼ばれる量です．
この積分を計算すれば，混合数 $K$ の事後分布を評価できるわけですが，潜在変数 $l^n$ およびパラメータ $\theta$ の積分となるため，解析的に解くのは困難なことがわかります．
そこで，MCMCを用いてこの計算を，**事後分布に従うサンプルの期待値** として計算することを考えます．

#### レプリカ交換法
上述した積分をサンプルの期待値として計算する手法としてレプリカ交換法（REMC）を紹介します．
REMCでは，$M$ 個のの $0=\beta_1<\beta_2<\cdots<\beta_M=1$ なるパラメータ（**逆温度**）を考えます．
ここで，逆温度 $\beta_m$ での目標分布を

$$
    p(l^n,\theta, K|x^n;\beta_m) \propto p(x^n,l^n|\theta)^{\beta_m}p(\theta|K) = \exp\\{-\beta_mE(\theta,l^n)\\}p(\theta|K)
$$

と設定します．
また，

$$
    z(\beta) = \iint p(l^n,\theta, K|x^n;\beta)\text{d}\theta\text{d}l^n=\iint \exp\\{-\beta E(\theta,l^n)\\}p(\theta|K)\text{d}\theta\text{d}l^n
$$

なる逆温度 $\beta$ の関数を考えます．
定義から，

$$
    z(\beta_1 = 0) = \iint p(\theta|K)\text{d}\theta\text{d}l^n = \int \text{d}l^n = K^n = C(n,K)
$$

となります．
また，

$$
    z(\beta_M = 1) = Z(K)
$$

であることもわかります．

$$
    \begin{aligned}
        Z(K)
        &= z(\beta_M) = C(n,K)\frac{z(\beta_M)}{z(\beta_1)} = C(n,K)\times\frac{z(\beta_2)}{z(\beta_1)}\times\frac{z(\beta_3)}{z(\beta_2)}\times\cdots\times\frac{z(\beta_M)}{z(\beta_{M-1})}\cr
        &= C(n,K)\prod_{m=1}^{M-1}\frac{z(\beta_{m+1})}{z(\beta_m)} = C(n,K)\prod_{m=1}^{M-1}\frac{\iint \exp\\{-\beta_{m+1} E(\theta,l^n)\\}p(\theta|K)\text{d}\theta\text{d}l^n}{\iint \exp\\{-\beta_m E(\theta,l^n)\\}p(\theta|K)\text{d}\theta\text{d}l^n}\cr
        &= C(n,K)\prod_{m=1}^{M-1}\Bigg\langle -(\beta_{m+1}-\beta_m)E(\theta, l^n)\Bigg\rangle_{q(\theta,l^n;\beta_m)}
    \end{aligned}
$$
