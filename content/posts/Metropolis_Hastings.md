---
title: メトロポリス法ってなんだし。
date: 2022-01-24
tags: [Bayesian Inference, Machine Learning, MCMC, Metropolis-Hastings]
description: "本記事では，マルコフ連鎖モンテカルロ法（Markov Chain Monte Carlo: MCMC）を実現する手法の１つである，メトロポリス法（Metropolis-Hastings Method）について解説してみます．また，数値実験を行います．"
math: true
image: "/image/profile.jpg"
---

本記事では，マルコフ連鎖モンテカルロ法（Markov Chain Monte Carlo: MCMC）を実現する手法の１つである，メトロポリス法（Metropolis-Hastings Method）について解説してみます．
また，簡単 of 簡単な数値実験を行います．

## メトロポリス法(Metropolis-Hastings Method)

## 数値実験
今回はメトロポリス法による1次元標準正規分布からのサンプリングを行います．
確率変数を $x\in\mathbb R$ とします．
標準正規分布の確率密度関数 $p(x)$ は

$$
    p(x|\mu = 0, \sigma^2 = 1) = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}x^2\right),
$$

となります．
現時点でのサンプルを $x$ とし，確率的摂動 $\delta\in\mathbb R$ を加えた，遷移先の候補点を $x' = x + \delta$とします．
メトロポリス法の更新則として，$x,x'$ を用いて

$$
    u = \min\left\\{1, \frac{p(x')}{p(x)}\right\\} = \min\left\\{1, \frac{\exp(-x^2/2)}{\exp(-x'^2/2)}\right\\}
$$

を計算し，これを遷移の採択確率とすれば良いです．
実際の計算においては，対数をとり，

$$
    \log u = \min\left\\{0, \frac{1}{2}(x^2 - x'^2)\right\\}
$$

を用いて計算すると実装が簡単になりますので，本記事ではこちらを用います．

MCMCにおいて，定常分布に落ち着くまで少し時間が必要であるため，最初にMCMCでのサンプリングを回しておく必要があります．
運動前のウォームアップのようなもので，**バーンイン**と呼ばれます．
このバーンイン期間に得られたサンプルは，通常，確率分布から得られたサンプルとは見做さずに捨ててしまいます．
バーンインのサンプル数をどのように決定するべきかという問題には未だ明確な解決策が示されていません．
この記事では，バーンインのサンプル数を $40,000$ としています．
バーンイン後に，$40,000$ サンプルを取得しました．

本実験に使用したコードを以下に掲載します．
コードには自作のライブラリ `numpy.hpp`（C++のvectorをPython の Numpy 形式で保存する関数群）と `xoshiro.h` （メルセンヌ・ツイスタよりも早い乱数生成エンジンのテンプレート）を使用しておりますが，これらについてはそれぞれ，[C++配列を.npy形式で保存したい癖の強いあなたへ。](../c++2npy)と[XorShiftという乱数生成法が早いらしい。](../xorshiro)という別記事を参照していただけると幸いです．

<details>
<summary>main.cpp</summary>

```C++:main.cpp
#include<bits/stdc++.h>
#include"include/numpy.hpp" //C++ vector を .npy 形式で保存するためのライブラリ
#include"include/xoshiro.h" //乱数生成エンジンのライブラリ

using namespace std;
using rng_type = xoshiro::rng128pp;

const int BURNIN_NUM = 40000; //バーンイン数
const int SAMPLE_NUM = 40000; //サンプリング数
const int seed = 2021; //乱数シード

/** ２乗を返す関数 */
inline float pow2(float x){
    return x * x;
}

/** 実験結果を保持する構造体 */
struct Result{
    
    //MCMCで得られたサンプルを格納するvector
    vector<float>sample;

    Result(){
        sample = vector<float>(SAMPLE_NUM);
    }

    //sampleを保存
    void dump(){
        SaveNpy("result/sample.npy", sample);
    }
};

/** Metropolis-Hastings法 の構造体 */
struct Metropolis_Hastings{
    
    //結果を格納する構造体
    Result result;

    //サンプルを表す変数 
    float sample;
    //ステップサイズの粒度を決定する変数
    float step_size;

    //乱数生成エンジン
    rng_type rng;
    
    //遷移の採択を決定する区間[0, 1]の一様分布に従う乱数
    uniform_real_distribution<float>dist;
    //確率的な摂動を表す区間[-1, 1]の一様分布に従う乱数
    uniform_real_distribution<float>step;

    /** コンストラクタ：各変数を初期化 */
    Metropolis_Hastings(){

        result = Result();

        rng = rng_type(seed);
        dist = uniform_real_distribution<float>(0.0f, 1.0f);
        step = uniform_real_distribution<float>(-1.0f, 1.0f);

        sample = dist(rng);
        step_size = 0.5;
    }

    //Matropolis-Hastings法によるMCMCを実行する関数
    void mcmc(){
        
        //閾値の対数
        float log_thr;

        //バーンインサンプリングを実行
        printf("mcmc: Start Burn in phase...\n");
        for(int i = 0; i < BURNIN_NUM; ++i){
            //確率的な摂動
            float delta = step_size * step(rng);
            //遷移の候補点
            float candidate = sample + delta;
            //閾値を計算
            log_thr = 0.5f * (pow2(sample) - pow2(candidate));
            //遷移を採択するのか判定
            if(log(dist(rng)) < log_thr){
                sample = candidate;
            }
        }

        //サンプリングを実行
        printf("mcmc: Start Sampling phase...\n");
        for(int i = 0; i < SAMPLE_NUM; ++i){
            //確率的な摂動
            float delta = step_size * step(rng);
            // 遷移の候補点
            float candidate = sample + delta;
            //閾値の計算
            log_thr = 0.5f * (pow2(sample) - pow2(candidate));
            //遷移を採択するのか判定
            if(log(dist(rng)) < log_thr){
                sample = candidate;
            }
            //結果の格納
            result.sample[i] = sample;
        }

        printf("mcmc: Done.\n");

        //結果を保存
        result.dump();

        printf("mcmc: Finish dumping samples.\n");
    }

};

int main(){
    Metropolis_Hastings model;
    model.mcmc();
    return 0;
}
```
</details>


それでは実験結果を見ていきましょう．
得られたサンプル系列 $x_1,x_2,\cdots, x_{40000}$ のヒストグラムを示します．

![](/image/Metropolis_Hastings_1.png)

図中赤線が，真の標準正規分布の確率密度関数を示します．
この結果を見ると，標準正規分布に従うサンプルが得られていることがわかりますね．

実験結果は`Python`の`matploylib`を用いて可視化しています．
以下に可視化に用いたコードを示します．

<details>
<summary>plot.py</summary>

```Python:plot.py
import numpy as np
from matplotlib import pyplot as plt

# 実験で得られたサンプリングを読み込む
sample = np.load("result/sample.npy")

# 適当に区間をとる
x = np.linspace(-4.0, 4.0, 300)
# 標準正規分布の確率密度関数
pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

# 以下，実験結果の描写
fig = plt.figure(figsize = (6, 4), dpi = 300)
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlim(-4.0, 4.0)
ax.set_ylim(0.0, 0.45)
ax.set_xlabel("$x$", fontsize = 16)
ax.set_ylabel("Probability Density", fontsize = 16)

# 標準正規分布の確率密度関数の描写
ax.plot(x, pdf, label = "Gaussian PDF", color = "red", lw = 2, alpha = 0.75)
ax.legend(fontsize = 14, loc = 1)

# 得られたサンプルのヒストグラムを描写
ax_ = ax.twinx()
ax_.hist(sample, bins = 100, alpha = 0.5, range = (-4.0, 4.0))
ax_.set_ylabel("Frequency", fontsize = 16)
```
</details>

### まとめ
本記事ではメトロポリス法について解説しました．
また，数値実験ではメトロポリス法によるMCMCで，標準正規分布からのサンプリングを行い，確率密度関数とともに可視化することにより，サンプリングがうまくいっていることが確認できました．
記事内では，確率分布からのサンプリングを目的としてメトロポリス法，MCMCを用いましたが，これは氷山の一角です．
メトロポリス法やMCMCは，機械学習における回帰問題，分類問題，クラスタリング，などにも応用することが可能です．
というか，こちらの方が実用的ですし，面白いです．
機械学習への応用はまた別の記事で書き連ねようと思いますので，よろしくお願いいたします．

##### 参考文献・参考記事
[Wikipedia：メトロポリス・ヘイスティングス法](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%83%88%E3%83%AD%E3%83%9D%E3%83%AA%E3%82%B9%E3%83%BB%E3%83%98%E3%82%A4%E3%82%B9%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0%E3%82%B9%E6%B3%95)