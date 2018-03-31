# tzm030321に、CVAEにコメントつけてみる by TFujishima
# 題材：tzm030329/Keras_VAE 
# 第一回でコメントしたDense,Input系の記述は既知としてコメント省略する

# ライブラリ定義: 詳細は省略: 実装部分で必要に応じてコメント
import numpy as np
import matplotlib as mpl
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import cm
from skimage import io as skio

import keras
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers.merge import concatenate

# メイン処理:2Dカラーマップ画像の準備
def main():
    sns.set_context('talk', font_scale=1.5)
    sns.set_style('white') # for image
    mpl.rc('image', cmap='inferno', interpolation='nearest') # for image
    pass

# CVAEのクラス
class CVAE():
    # コンストラクタ: デフォルトは784次元、中間層(nch)256次元
    def __init__(self):
        self.input_dim = 784
        self.latent_dim = 128
        nch = 256

        # input
	# 入力層の定義
	# 784次元の画像データ
        self.x_input = Input(shape=(self.input_dim,))
	# ラベル用のインプット10種類分
        self.label_input_enc = Input(shape=(10,))
        self.label_input_dec = Input(shape=(10,))

        # encoder
	# 活性化関数
	# 中間層定義:1層目画像側を256次元にする
        x = Dense(nch, activation='relu')(self.x_input)
	# 中間層定義:1層目ラベル側を256次元にする
        lx = Dense(nch, activation='relu')(self.label_input_enc)
	# 連結器でつなげて512次元に
        x = concatenate([x, lx])
	# その次の中間層で256次元にすることで、この256次元のパラメータが
        # 「画像から学習されたもの」か「ラベルベクトルから学習されたもの」かを
        # 切り分け不能にさせることで、ラベルベクトルと画像との関連付けが生まれる？
        # これは、あとでラベルベクトルで複数1に立たせることで中間のような画像も生成させることが狙い。
        x = Dense(nch, activation='relu')(x)
	# xを入力とした次の中間層をz_meanとz_log_varとする
        self.z_mean = Dense(self.latent_dim)(x)
        self.z_log_var = Dense(self.latent_dim)(x)
	# 後述のsampling関数にz_mean,z_log_varを入力したときの関数出力(出力次元はlatent_dim)をzに割り当てる
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # decoder
	# デコーダの書き方がエンコーダと違うのは、後で使うからか・・・
	# デコーダ：画像側、入力未定・出力nch=256次元のテンソルを関数型としてdec1に代入。
        self.dec1 = Dense(nch, activation='relu')
	# デコーダ：ラベル側も同様の考え方。
        self.dec1_label = Dense(nch, activation='relu')
	# 2層目を準備。dec2は必要なのか？dec1を繰り返し使ってはいけない？
        # やってみたが、この書き方は必要であることはモデルの描画機能を使えば見える。
        # 両方dec1でやってしまうと、ループ入力になる。
        # なお、モデルの描画機能は後日レポジトリに練習で組んだものをアップする予定。
        self.dec2 = Dense(nch, activation='relu')
	# デコーダ最終層値域を0~1にするためシグモイドを選択。
        self.dec_out = Dense(self.input_dim, activation='sigmoid')

        # ここから下は、代替処理を提案する。
	# デコーダ１層目の入力にzを割り当てる
        x = self.dec1(self.z)
	# 同じことなので説明省略
        lx = self.dec1_label(self.label_input_dec)
        x = concatenate([x, lx])
        x = self.dec2(x)
        self.x_out = self.dec_out(x)

        # ここまで

        # TFujishima：decoder層への入出力結合を関数化
        self.x_out = self.connect_decoder(self.z,self.label_input_dec)

    # サンプリングを定義。入力を偏差1.0のガウシアンでぼかす。
    def sampling(self, args):
        z_mean, z_log_var = args
        # ここは自信ない・・・z_meanは(batch,dim)の2階テンソルだったはずなので、
        # 入力バッチ数(定数をダイレクト入力しないのはバッチ数が不定だからこうしている？）
        nd = K.shape(z_mean)[0]
        # 入力次元
        nc = self.latent_dim
        # (バッチ、入力次元)の全テンソル成分についてランダムに平均0、stdev=1.0のノイズを引加
        eps = K.random_normal(shape=(nd, nc), mean=0., stddev=1.0)
        # 一様ノイズに分散の役割としたパラメータz_log_varで重みをかけて、VAEのぼかし処理を表現した？
        return z_mean + K.exp(z_log_var / 2) * eps

    def cvae(self):
        # 入力に[画像、学習用ラベル、デコーダ用ラベル]が入るようにして、モデル形成
        return Model([self.x_input, self.label_input_enc, self.label_input_dec], self.x_out)

    def encoder(self):
        # エンコーダだけならデコーダ入力はいらない
        return Model([self.x_input, self.label_input_enc], self.z_mean)

    def decoder(self):
        # デコーダ側は、符号化されたz値とdimが出せるようにする
        z = Input(shape=(self.latent_dim,))
        l = Input(shape=(10,))
        # ところで・・・
        # ↓ここから、
        x = self.dec1(z)
        lx = self.dec1_label(l)
        x = concatenate([x, lx])
        x = self.dec2(x)
        x_out = self.dec_out(x)
        # ↑ここまでを以下の関数化しても良いかも。
        return Model([z,l], self.connect_decoder(z,l))

    # TFujishima書き込み ここから
    def connect_decoder(self,inp):
        (z,l) = inp
        # z = Input(shape=(self.latent_dim,))
        # l = Input(shape=(10,))
        x = self.dec1(z)
        lx = self.dec1_label(l)
        x = concatenate([x, lx])
        x = self.dec2(x)
        x_out = self.dec_out(x)
    # ここまで

    # 損失関数の定義
    def loss(self):
        # 画像側の類似度評価はcrossentropyを使う
        bce = metrics.binary_crossentropy(self.x_input, self.x_out)
        # ラベル側の損失関数と足すので、ラベル側の振幅を画像の入力次元だけ増幅しておく。
        xent_loss = self.input_dim * bce
        # 平均値が大きいほど良い解。z_log_varは全体的に小さく、指数演算では大きいほどよい。
        # これでz_log_varの寄与度にメリハリをつける効果を狙ったっぽい。
        kl = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        # 最終軸成分(つまり入力次元分)の和を取ってラベル側の損失関数を生成
        # 最初の軸成分はバッチ処理に使うサンプル数が格納されているので、ここで和を取らない
        kl_loss = - 0.5 * K.sum(kl, axis=-1)
        # 画像側とラベル側の損失関数を平均して最終的な損失評価値を生成
        # バッチ処理のためにサンプル数分を平均化して最終的な損失値が決定
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss


if __name__ == '__main__':
    main()

    seed = 0 # for network init

    batch_size = 32
    epochs = 2000
    nd = 1000 # number of images for training
    ndo = 100 # number of images for testing

    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    x_train = x_train.astype('float32') / 255.
    x_train = x_train[:nd]
    x_train = x_train.reshape(nd, -1)
    y_train = y_train[:nd]
    # y_train,y_testは画像に対応した番号値リスト(0～9)。
    # np.eye(10)は単位行列でありnp.eye(10)[0]なら[1.,0.,0.,....]となるので、
    # 以下のように設定すればy_trainの番号に対応した列の値が1、そうでなければ0の
    # 10次元ベクトル x サンプル数の配列が生成される
    y_train = np.eye(10)[y_train] # to one-hot

    x_test = x_test.astype('float32') / 255.
    x_test = x_test[:ndo]
    x_test = x_test.reshape(ndo, -1)
    y_test = y_test[:ndo]
    y_test = np.eye(10)[y_test] # to one-hot

    # make vae model and train
    np.random.seed(seed)
    # CVAEをインスタンス
    cvae = CVAE()
    # モデルを生成する
    model = cvae.cvae()
    model.summary()
    # 損失関数の追加
    model.add_loss(cvae.loss())
    np.random.seed(seed)
    # adamで最適化。
    model.compile(optimizer='adam', loss=None)
    # あとはいいや・・・
    model.fit([x_train, y_train, y_train], epochs=epochs, batch_size=batch_size)

    # extract latent z
    pos = [] # select nu
    for i in range(10):
        n = x_train.shape[0]
        mask = (np.argmax(y_train, axis=-1)==i)
        pos.append(np.arange(n)[mask][0])

    x = x_train[pos]
    y = y_train[pos]
    enc = cvae.encoder()
    z = enc.predict([x,y])

    # style gen
    zz = z.repeat(10, axis=0)
    yy = np.tile(np.eye(10), 10).T
    dec = cvae.decoder()
    xx = dec.predict([zz, yy])
    xin = x.reshape(280,28)
    xin = np.append(xin, np.zeros((280,5))+1.0, axis=1)
    xout = xx.reshape(10,10,28,28).transpose(3,0,2,1).reshape(280,280)
    dst = np.append(xin, xout, axis=1)

    # plot images
    plt.imshow(dst)
    plt.tight_layout();plt.show()

    # output images
    out = (cm.inferno(dst)[:,:,:3]*255).astype(np.uint8)
    out[:,28:33] = 255
    skio.imsave('png/cvae_out_%03d.png' % seed, out)

