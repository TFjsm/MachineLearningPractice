from keras.datasets import mnist 
import numpy as np


from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
import keras.regularizers as regularizers
import matplotlib.pyplot as plt

def RunAsMain():
	ae = Design_DefaultAutoEncoder()
	res = Train(ae,5)
	ShowImgs(res)
	pass

def Design_DefaultAutoEncoder():
	encoding_dim = 32
	input_img = Input(shape=(784,))
	encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
	decoded = Dense(784, activation='sigmoid')(encoded)
	autoencoder = Model(input_img, decoded)
	encoder = Model(input_img, encoded)
	encoded_input = Input(shape=(encoding_dim,))
	decoder_layer = autoencoder.layers[-1]
	decoder = Model(encoded_input, decoder_layer(encoded_input))
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	return (autoencoder,encoder,decoder)

def Train(RefModel,epoch):
	(autoencoder,encoder,decoder) = RefModel
	(x_train, _), (x_test, _) = mnist.load_data()
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	autoencoder.fit(x_train, x_train,
		epochs=epoch,
		batch_size=256,
		shuffle=True,
		validation_data=(x_test, x_test))
	encoded_imgs = encoder.predict(x_test)
	decoded_imgs = decoder.predict(encoded_imgs)
	n = 10
	return (np.array((x_test[:n**2],decoded_imgs[:n**2])),n)

def ShowImgs(RefImgs):
	(content,n) = RefImgs
	img = content.reshape(2*n, n, 28, 28).transpose(0,2,1,3).reshape(n*28*2, n*28)
	plt.imshow(img)
	plt.show()

if __name__ == '__main__':
	RunAsMain()
