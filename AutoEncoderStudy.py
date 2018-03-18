# Kerasを使用したオートエンコーダ訓練をmnistデータセットで行うスクリプトを説明してみる
# by TFujishima
# 題材：https://github.com/evgenyorlov1/AutoEncoder/sparse.py

# 残存課題：
# ・下記説明、Model第一項の説明がちょっと怪しい
# この人は他にDense, Deep, CAEをアップしている。
# CAEは下記概念を積層してMaxPoolingとComvolutionを繰り返しているような感じだった。
# Deepは、Denseを何層か重ねている感じだったので、下記の延長という感じ。
# Denseは見てない。

# 定義は説明省略
from keras.datasets import mnist 
import numpy as np

# データセットのロード部

# mnistデータセットから文字データをロードする
# ロードされるデータは以下。
# x_train: 1サンプルあたり28x28の２次元配列、値域0~255の整数値が60000サンプル
# x_test:  1サンプルあたり28x28の２次元配列、値域0~255の整数値が10000サンプル
(x_train, _), (x_test, _) = mnist.load_data()

# 訓練用データx_train,テストデータx_testをそれぞれfloat32に変換して、255で割る
# 割った後のデータは値域[0.0,1.0]となる
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# reshape関数によって
# list.shape関数で２次元配列長(28,28)をtupleで取得。
# np.prod関数でtupleの積を得て、(サンプル数,28x28次元)のデータ列にreshapeする
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#### MNISTではなく自由画像を使う場合のアプローチ想定 ####
# ロード部は画像データをN(今回の場合はN=784)次元ベクトルに直す変換さえすればよいので、
# 代わりにSkImage経由でJPG/PNGからロードして、NumpyでKerasに入力できる上記形状の配列に変形してもいい。
# ただし、カラー画像入力の場合はRGBのチャネルのうちどれか一つを選ぶか、YCbCrに変換したうえで
# Yのチャネルのみを持ってこないと、学習できないはず。
##########################################################

###################
# ここから本題
# オートエンコーダ構築

# エンコードした後の次元を32次元にする
encoding_dim = 32

# 入力層に28x28次元が入るようにする
input_img = Input(shape=(784,))

# エンコーダで32次元にする(入力層-エンコーダまでの間に中間層なし)
# 活性化関数はランプ関数、L1正則化を適用。→L1なので毎回に重みに対して10e-5の0になるようにバイアスがかかる。
# エンコード部は特徴量獲得のためL1を使用していると思われる(L2は回帰目的で使うのが良い気がするので・・・）。
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)

# シグモイドを使って元の次元に戻そうとする層を作り、Denseで作ったレイヤーの入力にencoded出力をつなげる
# これをdecodedと定義する
# シグモイドを使っている理由は、decoded部は画像生成用途なので、だろうか？
decoded = Dense(784, activation='sigmoid')(encoded)

# 以上、784->32->784の３層のオートエンコーダを構成し、input_imgから入力する
# decodedは、上記指定によって必然的に3層持っている。
autoencoder = Model(input_img, decoded)

# エンコーダとデコーダの動きが見えるようにしてみる
# これには、encodedまでのレイヤーの出力を最終として、input_imgから入力することを明記すればよい。
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))

# オートエンコーダ最終(出力)層をインスタンス(この解釈はちょっと怪しい・・・後で調べる)
# これはデコーダ側が1層だけだから成り立つ記述かもしれない
decoder_layer = autoencoder.layers[-1]

# エンコードデータを入力とし、エンコードデータを入力に繋げた最終層で出力すれば、
# エンコード入力に対応した画像を生成するデコーダになる
# decoderは出力のテストのために使用する
decoder = Model(encoded_input, decoder_layer(encoded_input))

# オートエンコーダを学習できるように準備する
# crossentropyを損失評価に使って最適解はadadeltaで探索する
# 交差エントロピーを使っているのはなぜだろう？
# 合成画像と元画像の差分が小さくなるように評価関数を作るなら、損失評価関数は自己相関関数に似ているhingeの方が良さそうに見えた・・・。
# ちなみに、この作者のCAEサンプルではhingeを使っている。
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



# 訓練(50フェーズ、1回の損失評価あたり256個のバッチ学習。訓練データはシャッフルして使う。
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# 訓練結果を画像化するのはこれでいけるらしい。
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# ここまで本題
#################

# 標準的な表示操作
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
