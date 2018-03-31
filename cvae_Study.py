# tzm030321�ɁACVAE�ɃR�����g���Ă݂� by TFujishima
# ��ށFtzm030329/Keras_VAE 
# ����ŃR�����g����Dense,Input�n�̋L�q�͊��m�Ƃ��ăR�����g�ȗ�����

# ���C�u������`: �ڍׂ͏ȗ�: ���������ŕK�v�ɉ����ăR�����g
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

# ���C������:2D�J���[�}�b�v�摜�̏���
def main():
    sns.set_context('talk', font_scale=1.5)
    sns.set_style('white') # for image
    mpl.rc('image', cmap='inferno', interpolation='nearest') # for image
    pass

# CVAE�̃N���X
class CVAE():
    # �R���X�g���N�^: �f�t�H���g��784�����A���ԑw(nch)256����
    def __init__(self):
        self.input_dim = 784
        self.latent_dim = 128
        nch = 256

        # input
	# ���͑w�̒�`
	# 784�����̉摜�f�[�^
        self.x_input = Input(shape=(self.input_dim,))
	# ���x���p�̃C���v�b�g10��ޕ�
        self.label_input_enc = Input(shape=(10,))
        self.label_input_dec = Input(shape=(10,))

        # encoder
	# �������֐�
	# ���ԑw��`:1�w�ډ摜����256�����ɂ���
        x = Dense(nch, activation='relu')(self.x_input)
	# ���ԑw��`:1�w�ڃ��x������256�����ɂ���
        lx = Dense(nch, activation='relu')(self.label_input_enc)
	# �A����łȂ���512������
        x = concatenate([x, lx])
	# ���̎��̒��ԑw��256�����ɂ��邱�ƂŁA����256�����̃p�����[�^��
        # �u�摜����w�K���ꂽ���́v���u���x���x�N�g������w�K���ꂽ���́v����
        # �؂蕪���s�\�ɂ����邱�ƂŁA���x���x�N�g���Ɖ摜�Ƃ̊֘A�t�������܂��H
        # ����́A���ƂŃ��x���x�N�g���ŕ���1�ɗ������邱�ƂŒ��Ԃ̂悤�ȉ摜�����������邱�Ƃ��_���B
        x = Dense(nch, activation='relu')(x)
	# x����͂Ƃ������̒��ԑw��z_mean��z_log_var�Ƃ���
        self.z_mean = Dense(self.latent_dim)(x)
        self.z_log_var = Dense(self.latent_dim)(x)
	# ��q��sampling�֐���z_mean,z_log_var����͂����Ƃ��̊֐��o��(�o�͎�����latent_dim)��z�Ɋ��蓖�Ă�
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # decoder
	# �f�R�[�_�̏��������G���R�[�_�ƈႤ�̂́A��Ŏg�����炩�E�E�E
	# �f�R�[�_�F�摜���A���͖���E�o��nch=256�����̃e���\�����֐��^�Ƃ���dec1�ɑ���B
        self.dec1 = Dense(nch, activation='relu')
	# �f�R�[�_�F���x���������l�̍l�����B
        self.dec1_label = Dense(nch, activation='relu')
	# 2�w�ڂ������Bdec2�͕K�v�Ȃ̂��Hdec1���J��Ԃ��g���Ă͂����Ȃ��H
        # ����Ă݂����A���̏������͕K�v�ł��邱�Ƃ̓��f���̕`��@�\���g���Ό�����B
        # ����dec1�ł���Ă��܂��ƁA���[�v���͂ɂȂ�B
        # �Ȃ��A���f���̕`��@�\�͌�����|�W�g���ɗ��K�őg�񂾂��̂��A�b�v����\��B
        self.dec2 = Dense(nch, activation='relu')
	# �f�R�[�_�ŏI�w�l���0~1�ɂ��邽�߃V�O���C�h��I���B
        self.dec_out = Dense(self.input_dim, activation='sigmoid')

        # �������牺�́A��֏������Ă���B
	# �f�R�[�_�P�w�ڂ̓��͂�z�����蓖�Ă�
        x = self.dec1(self.z)
	# �������ƂȂ̂Ő����ȗ�
        lx = self.dec1_label(self.label_input_dec)
        x = concatenate([x, lx])
        x = self.dec2(x)
        self.x_out = self.dec_out(x)

        # �����܂�

        # TFujishima�Fdecoder�w�ւ̓��o�͌������֐���
        self.x_out = self.connect_decoder(self.z,self.label_input_dec)

    # �T���v�����O���`�B���͂�΍�1.0�̃K�E�V�A���łڂ����B
    def sampling(self, args):
        z_mean, z_log_var = args
        # �����͎��M�Ȃ��E�E�Ez_mean��(batch,dim)��2�K�e���\���������͂��Ȃ̂ŁA
        # ���̓o�b�`��(�萔���_�C���N�g���͂��Ȃ��̂̓o�b�`�����s�肾���炱�����Ă���H�j
        nd = K.shape(z_mean)[0]
        # ���͎���
        nc = self.latent_dim
        # (�o�b�`�A���͎���)�̑S�e���\�������ɂ��ă����_���ɕ���0�Astdev=1.0�̃m�C�Y������
        eps = K.random_normal(shape=(nd, nc), mean=0., stddev=1.0)
        # ��l�m�C�Y�ɕ��U�̖����Ƃ����p�����[�^z_log_var�ŏd�݂������āAVAE�̂ڂ���������\�������H
        return z_mean + K.exp(z_log_var / 2) * eps

    def cvae(self):
        # ���͂�[�摜�A�w�K�p���x���A�f�R�[�_�p���x��]������悤�ɂ��āA���f���`��
        return Model([self.x_input, self.label_input_enc, self.label_input_dec], self.x_out)

    def encoder(self):
        # �G���R�[�_�����Ȃ�f�R�[�_���͂͂���Ȃ�
        return Model([self.x_input, self.label_input_enc], self.z_mean)

    def decoder(self):
        # �f�R�[�_���́A���������ꂽz�l��dim���o����悤�ɂ���
        z = Input(shape=(self.latent_dim,))
        l = Input(shape=(10,))
        # �Ƃ���ŁE�E�E
        # ����������A
        x = self.dec1(z)
        lx = self.dec1_label(l)
        x = concatenate([x, lx])
        x = self.dec2(x)
        x_out = self.dec_out(x)
        # �������܂ł��ȉ��̊֐������Ă��ǂ������B
        return Model([z,l], self.connect_decoder(z,l))

    # TFujishima�������� ��������
    def connect_decoder(self,inp):
        (z,l) = inp
        # z = Input(shape=(self.latent_dim,))
        # l = Input(shape=(10,))
        x = self.dec1(z)
        lx = self.dec1_label(l)
        x = concatenate([x, lx])
        x = self.dec2(x)
        x_out = self.dec_out(x)
    # �����܂�

    # �����֐��̒�`
    def loss(self):
        # �摜���̗ގ��x�]����crossentropy���g��
        bce = metrics.binary_crossentropy(self.x_input, self.x_out)
        # ���x�����̑����֐��Ƒ����̂ŁA���x�����̐U�����摜�̓��͎��������������Ă����B
        xent_loss = self.input_dim * bce
        # ���ϒl���傫���قǗǂ����Bz_log_var�͑S�̓I�ɏ������A�w�����Z�ł͑傫���قǂ悢�B
        # �����z_log_var�̊�^�x�Ƀ����n����������ʂ�_�������ۂ��B
        kl = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        # �ŏI������(�܂���͎�����)�̘a������ă��x�����̑����֐��𐶐�
        # �ŏ��̎������̓o�b�`�����Ɏg���T���v�������i�[����Ă���̂ŁA�����Řa�����Ȃ�
        kl_loss = - 0.5 * K.sum(kl, axis=-1)
        # �摜���ƃ��x�����̑����֐��𕽋ς��čŏI�I�ȑ����]���l�𐶐�
        # �o�b�`�����̂��߂ɃT���v�������𕽋ω����čŏI�I�ȑ����l������
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
    # y_train,y_test�͉摜�ɑΉ������ԍ��l���X�g(0�`9)�B
    # np.eye(10)�͒P�ʍs��ł���np.eye(10)[0]�Ȃ�[1.,0.,0.,....]�ƂȂ�̂ŁA
    # �ȉ��̂悤�ɐݒ肷���y_train�̔ԍ��ɑΉ�������̒l��1�A�����łȂ����0��
    # 10�����x�N�g�� x �T���v�����̔z�񂪐��������
    y_train = np.eye(10)[y_train] # to one-hot

    x_test = x_test.astype('float32') / 255.
    x_test = x_test[:ndo]
    x_test = x_test.reshape(ndo, -1)
    y_test = y_test[:ndo]
    y_test = np.eye(10)[y_test] # to one-hot

    # make vae model and train
    np.random.seed(seed)
    # CVAE���C���X�^���X
    cvae = CVAE()
    # ���f���𐶐�����
    model = cvae.cvae()
    model.summary()
    # �����֐��̒ǉ�
    model.add_loss(cvae.loss())
    np.random.seed(seed)
    # adam�ōœK���B
    model.compile(optimizer='adam', loss=None)
    # ���Ƃ͂�����E�E�E
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

