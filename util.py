import tensorflow as tf

def mish(x,name):
    return x * tf.nn.tanh( tf.nn.softplus(x),name=name)

def upsample(x):
    return tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear')

def conv2d(x,filter,kernel,stride=1,name=None,activation='mish'):
    x = tf.keras.layers.Conv2D(filter,kernel,stride,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation=='mish':
        return mish(x,name)
    elif activation=='leaky':
        return tf.keras.layers.LeakyReLU(name=name)(x)

def convset(x, filter):
    x = conv2d(x, filter, 1,activation='leaky')
    x = conv2d(x, filter * 2, 3,activation='leaky')
    x = conv2d(x, filter, 1,activation='leaky')
    x = conv2d(x, filter * 2, 3,activation='leaky')
    return conv2d(x, filter, 1,activation='leaky')