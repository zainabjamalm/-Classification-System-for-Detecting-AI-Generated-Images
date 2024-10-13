import tensorflow as tf
from tensorflow.keras import Model, layers

NO_CAPS = 10
@tf.keras.utils.register_keras_serializable()
class StatsPooling(layers.Layer):
    def __init__(self, **kwargs):
        super(StatsPooling, self).__init__(**kwargs)
    
    def call(self, x):
        shape = tf.shape(x)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, (batch_size, channels, width * height))
        mean = tf.reduce_mean(x, axis=-1)
        std = tf.math.reduce_std(x, axis=-1)
        return tf.stack([mean, std], axis=-1)
    
    def get_config(self):
        config = super(StatsPooling, self).get_config()
        return config
@tf.keras.utils.register_keras_serializable()
class View(layers.Layer):
    def __init__(self, shape, **kwargs):
        super(View, self).__init__(**kwargs)
        self.shape = shape
        
    def call(self, input):
        return tf.reshape(input, self.shape)
    
    def get_config(self):
        config = super(View, self).get_config()
        config.update({'shape': self.shape})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
@tf.keras.utils.register_keras_serializable()
class VggExtractor(Model):
    def __init__(self, train=False, **kwargs):
        super(VggExtractor, self).__init__(**kwargs)
        self.vgg_1 = self.Vgg(tf.keras.applications.VGG19(include_top=False, weights='imagenet'), 0, 11)
        self.train_mode = train
        self.vgg_1.trainable = train

    def Vgg(self, vgg, begin, end):
        model = Model(inputs=vgg.input, outputs=vgg.layers[end].output)
        return model

    def call(self, input):
        return self.vgg_1(input)

    def get_config(self):
        config = super(VggExtractor, self).get_config()
        config.update({'train': self.train_mode})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
@tf.keras.utils.register_keras_serializable()
class PrimaryCapsule(Model):
    def __init__(self, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.capsules = [self.create_capsule() for _ in range(NO_CAPS)]

    def create_capsule(self):
        return tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same', input_shape=(None, None, 256)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(16, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Lambda(lambda x: StatsPooling()(x)),
            layers.Conv1D(8, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(1, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            View((-1, 8))
        ])

    def squash(self, tensor, axis):
        squared_norm = tf.reduce_sum(tf.square(tensor), axis=axis, keepdims=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / tf.sqrt(squared_norm)

    def call(self, x):
        outputs = [capsule(x) for capsule in self.capsules]
        output = tf.stack(outputs, axis=-1)
        return self.squash(output, axis=-1)
    
    def get_config(self):
        config = super(PrimaryCapsule, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
@tf.keras.utils.register_keras_serializable()
class RoutingLayer(layers.Layer):
    def __init__(self, num_input_capsules, num_output_capsules, data_in, data_out, num_iterations, **kwargs):
        super(RoutingLayer, self).__init__(**kwargs)
        self.num_iterations = num_iterations
        self.num_input_capsules = num_input_capsules
        self.num_output_capsules = num_output_capsules
        self.data_in = data_in
        self.data_out = data_out
        self.route_weights = self.add_weight(shape=[1, num_input_capsules, num_output_capsules, data_in, data_out],
                                             initializer='random_normal',
                                             trainable=True)

    def squash(self, tensor, axis=-1):
        squared_norm = tf.reduce_sum(tf.square(tensor), axis=axis, keepdims=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / tf.sqrt(squared_norm)

    def call(self, u, random=False, dropout=0.0):
        batch_size = tf.shape(u)[0]
        u = tf.reshape(u, shape=[batch_size, self.num_input_capsules, 1, self.data_in, 1])
        u = tf.tile(u, [1, 1, self.num_output_capsules, 1, 1])

        if random:
            noise = tf.random.normal(self.route_weights.shape, stddev=0.01)
            route_weights = self.route_weights + noise
        else:
            route_weights = self.route_weights

        u_hat = tf.matmul(route_weights, u, transpose_a=True)
        u_hat = tf.reshape(u_hat, [batch_size, self.num_input_capsules, self.num_output_capsules, self.data_out])

        if dropout > 0.0:
            u_hat = tf.nn.dropout(u_hat, rate=dropout)

        b = tf.zeros_like(u_hat[:, :, :, 0])

        for i in range(self.num_iterations):
            c = tf.nn.softmax(b, axis=2)
            outputs = self.squash(tf.reduce_sum(c[:, :, :, tf.newaxis] * u_hat, axis=1))
            if i != self.num_iterations - 1:
                b += tf.reduce_sum(u_hat * outputs[:, tf.newaxis, :, :], axis=-1)

        return outputs
    
    def get_config(self):
        config = super(RoutingLayer, self).get_config()
        config.update({
            'num_iterations': self.num_iterations,
            'num_input_capsules': self.num_input_capsules,
            'num_output_capsules': self.num_output_capsules,
            'data_in': self.data_in,
            'data_out': self.data_out
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
@tf.keras.utils.register_keras_serializable()
class OutputCapsule(layers.Layer):
    def call(self, inputs):
        classes = tf.nn.softmax(inputs, axis=1)
        class_mean = tf.reduce_mean(classes, axis=-1)
        return class_mean

    def get_config(self):
        config = super(OutputCapsule, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
