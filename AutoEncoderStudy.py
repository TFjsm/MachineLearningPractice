# Keras���g�p�����I�[�g�G���R�[�_�P����mnist�f�[�^�Z�b�g�ōs���X�N���v�g��������Ă݂�
# by TFujishima
# ��ށFhttps://github.com/evgenyorlov1/AutoEncoder/sparse.py

# �c���ۑ�F
# �E���L�����AModel��ꍀ�̐�����������Ɖ�����
# ���̐l�͑���Dense, Deep, CAE���A�b�v���Ă���B
# CAE�͉��L�T�O��ϑw����MaxPooling��Comvolution���J��Ԃ��Ă���悤�Ȋ����������B
# Deep�́ADense�����w���d�˂Ă��銴���������̂ŁA���L�̉����Ƃ��������B
# Dense�͌��ĂȂ��B

# ��`�͐����ȗ�
from keras.datasets import mnist 
import numpy as np

# �f�[�^�Z�b�g�̃��[�h��

# mnist�f�[�^�Z�b�g���當���f�[�^�����[�h����
# ���[�h�����f�[�^�͈ȉ��B
# x_train: 1�T���v��������28x28�̂Q�����z��A�l��0~255�̐����l��60000�T���v��
# x_test:  1�T���v��������28x28�̂Q�����z��A�l��0~255�̐����l��10000�T���v��
(x_train, _), (x_test, _) = mnist.load_data()

# �P���p�f�[�^x_train,�e�X�g�f�[�^x_test�����ꂼ��float32�ɕϊ����āA255�Ŋ���
# ��������̃f�[�^�͒l��[0.0,1.0]�ƂȂ�
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# reshape�֐��ɂ����
# list.shape�֐��łQ�����z��(28,28)��tuple�Ŏ擾�B
# np.prod�֐���tuple�̐ς𓾂āA(�T���v����,28x28����)�̃f�[�^���reshape����
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#### MNIST�ł͂Ȃ����R�摜���g���ꍇ�̃A�v���[�`�z�� ####
# ���[�h���͉摜�f�[�^��N(����̏ꍇ��N=784)�����x�N�g���ɒ����ϊ���������΂悢�̂ŁA
# �����SkImage�o�R��JPG/PNG���烍�[�h���āANumpy��Keras�ɓ��͂ł����L�`��̔z��ɕό`���Ă������B
# �������A�J���[�摜���͂̏ꍇ��RGB�̃`���l���̂����ǂꂩ���I�Ԃ��AYCbCr�ɕϊ�����������
# Y�̃`���l���݂̂������Ă��Ȃ��ƁA�w�K�ł��Ȃ��͂��B
##########################################################

###################
# ��������{��
# �I�[�g�G���R�[�_�\�z

# �G���R�[�h������̎�����32�����ɂ���
encoding_dim = 32

# ���͑w��28x28����������悤�ɂ���
input_img = Input(shape=(784,))

# �G���R�[�_��32�����ɂ���(���͑w-�G���R�[�_�܂ł̊Ԃɒ��ԑw�Ȃ�)
# �������֐��̓����v�֐��AL1��������K�p�B��L1�Ȃ̂Ŗ���ɏd�݂ɑ΂���10e-5��0�ɂȂ�悤�Ƀo�C�A�X��������B
# �G���R�[�h���͓����ʊl���̂���L1���g�p���Ă���Ǝv����(L2�͉�A�ړI�Ŏg���̂��ǂ��C������̂ŁE�E�E�j�B
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)

# �V�O���C�h���g���Č��̎����ɖ߂����Ƃ���w�����ADense�ō�������C���[�̓��͂�encoded�o�͂��Ȃ���
# �����decoded�ƒ�`����
# �V�O���C�h���g���Ă��闝�R�́Adecoded���͉摜�����p�r�Ȃ̂ŁA���낤���H
decoded = Dense(784, activation='sigmoid')(encoded)

# �ȏ�A784->32->784�̂R�w�̃I�[�g�G���R�[�_���\�����Ainput_img������͂���
# decoded�́A��L�w��ɂ���ĕK�R�I��3�w�����Ă���B
autoencoder = Model(input_img, decoded)

# �G���R�[�_�ƃf�R�[�_�̓�����������悤�ɂ��Ă݂�
# ����ɂ́Aencoded�܂ł̃��C���[�̏o�͂��ŏI�Ƃ��āAinput_img������͂��邱�Ƃ𖾋L����΂悢�B
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))

# �I�[�g�G���R�[�_�ŏI(�o��)�w���C���X�^���X(���̉��߂͂�����Ɖ������E�E�E��Œ��ׂ�)
# ����̓f�R�[�_����1�w���������琬�藧�L�q��������Ȃ�
decoder_layer = autoencoder.layers[-1]

# �G���R�[�h�f�[�^����͂Ƃ��A�G���R�[�h�f�[�^����͂Ɍq�����ŏI�w�ŏo�͂���΁A
# �G���R�[�h���͂ɑΉ������摜�𐶐�����f�R�[�_�ɂȂ�
# decoder�͏o�͂̃e�X�g�̂��߂Ɏg�p����
decoder = Model(encoded_input, decoder_layer(encoded_input))

# �I�[�g�G���R�[�_���w�K�ł���悤�ɏ�������
# crossentropy�𑹎��]���Ɏg���čœK����adadelta�ŒT������
# �����G���g���s�[���g���Ă���̂͂Ȃ����낤�H
# �����摜�ƌ��摜�̍������������Ȃ�悤�ɕ]���֐������Ȃ�A�����]���֐��͎��ȑ��֊֐��Ɏ��Ă���hinge�̕����ǂ������Ɍ������E�E�E�B
# ���Ȃ݂ɁA���̍�҂�CAE�T���v���ł�hinge���g���Ă���B
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



# �P��(50�t�F�[�Y�A1��̑����]��������256�̃o�b�`�w�K�B�P���f�[�^�̓V���b�t�����Ďg���B
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# �P�����ʂ��摜������̂͂���ł�����炵���B
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# �����܂Ŗ{��
#################

# �W���I�ȕ\������
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
