# Q.5 Wasserstein GAN (WGAN)

## what is mode collapse?
学習が不十分な識別器(discreminator)に対して、生成器(generator)を最適してしまったなどで複雑な実世界のデータ分布を表すことを学ぶことができず、生成器から生成される画像がある特定のクラスに集中してしまい、データセットの多様性が失われてしまったり、新しいデータへの一般化が不十分になる現象。
<br>[From GAN to WGAN (2019)](https://arxiv.org/abs/1904.08994)
<br>モード崩壊を防ぐ方法として生成器のエントロピーを最大化する手法などがあるみたい。
<br>[Prescribed Generative Adversarial Networks (2019)](https://arxiv.org/abs/1910.04302)

## Problem with BCE loss
logの中が0になると計算不可能となり正しく計算が行われなくなってしまう。GANでBCE Lossを用いる場合、logの中が0になる可能性が存在するため、正しく学習が行われずに勾配消失等の問題が発生してしまう。
<br>![スクリーンショット 2021-10-21 19 45 38](https://user-images.githubusercontent.com/64674323/138264730-731bc38c-5e33-4710-b9ee-41d980e47782.png)
<br>[Wasserstein GAN (2017)](https://arxiv.org/abs/1701.07875)
## Wasserstein loss
多くのGANでは損失関数としてJensen-Shannon divergenceを指標とした関数(BCE loss)が用いられているが、このような損失関数を利用する場合、問題点として勾配消失問題が存在する。そのため、WGANではWasserstein距離を指標として用いており、最適点付近で勾配が消失してしまうのを防いでいる。
<br> →Wasserstein lossはWasserstein距離を用いた損失関数
### Wasserstein距離を求めるためにKantorovich-Rubinstein dualityを利用
<br>![スクリーンショット 2021-10-21 19 41 48](https://user-images.githubusercontent.com/64674323/138264681-26c9c815-0916-44f4-9d5d-9bd4c43bb608.png)
### fを1-Lipschitzを満たす関数に置き換えるとlogのないBCE lossのような形に変形できる
<br>![スクリーンショット 2021-10-21 19 43 48](https://user-images.githubusercontent.com/64674323/138264697-6e12d349-d443-45af-b226-a4a8f2b04b20.png)
### Jensen-Shannon divergenceを用いた損失関数 (BCE)
<br>![スクリーンショット 2021-10-21 19 45 38](https://user-images.githubusercontent.com/64674323/138264730-731bc38c-5e33-4710-b9ee-41d980e47782.png)
<br>[Wasserstein GAN (2017)](https://arxiv.org/abs/1701.07875)

## Conditions of Wasserstein’s critic and 1-Lipschitz continuity enforcement
1-Lipschitz制約を適応するために、Wasserstein’s criticの重みを[-c,c]に収まるようにクリッピングを行う。

<br>![スクリーンショット 2021-10-21 18 11 21](https://user-images.githubusercontent.com/64674323/138265271-55469d03-4c33-4c27-89ef-6bf06e2b4ed4.png)
<br> → weight clippingでは重みが二極化してしまい、勾配爆発や消失が起きてしまう問題点があった。
<br>Gulrajaniらによって、WGANの最適化されたCriticが生成データの分布,訓練データの分布下のほぼ全ての点において単位勾配ノルムを持つことが証明されたため、これを満たすようにペナルティ項を導入することで上記の問題を解決したWGAN-GPというものが提案されている。
<br>[Improved Training of Wasserstein GANs (2017)](https://arxiv.org/abs/1704.00028)
