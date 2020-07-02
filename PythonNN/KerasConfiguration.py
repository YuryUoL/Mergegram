from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras import layers


class Linear(layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


### Our coeffdsicent layer:

class GridLayer(layers.Layer):
    def __init__(self, gridsize, grid_bnds = ((0.001, 1.001),(-0.001,1.001)), gridInit = tf.random_uniform_initializer(1.0,1.0)):
        super(GridLayer, self).__init__()
        self.gridsize = gridsize
        self.grid_bnds = grid_bnds
        self.gridInit = gridInit

    def build(self, input_shape):
        iv = self.gridInit(shape=self.gridsize, dtype='float32')
        self.W = tf.Variable(iv, trainable=True)

    def call(self, input):
        dimension_diag = input.shape[2]
        indices = []
        for dim in range(dimension_diag):
            [m, M] = self.grid_bnds[dim]
            coords = tf.slice(input, [0, 0, dim], [-1, -1, 1])
            ids = self.gridsize[dim] * (coords - m) / (M - m)
            indices.append(tf.cast(ids, tf.int32))

        return tf.expand_dims(tf.gather_nd(params=self.W, indices=tf.concat(indices, axis=2)), -1)

class TrivialCoefficent(layers.Layer):
    def __init__(self, coefficent = tf.ones):
        super(TrivialCoefficent, self).__init__()
        self.coefficent = coefficent

    def build(self, input_shape):
        """
        Ok
        :param input_shape:
        """
        #abc = tf.random_uniform_initializer(1.0,1.0)
        b = input_shape[1]
        iv = self.coefficent(shape=1, dtype='float32')
        self.w = tf.Variable(initial_value=iv,
                             trainable=True)

    def call(self, inputs):
        return inputs[:, :, 1:2] - inputs[:, :, 1:2] + self.w


class CoefficentLDDL(layers.Layer):
    """
    This is linear coefficient layer
    """
    def __init__(self, trainable = True, coefficient=tf.ones):
        super(CoefficentLDDL, self).__init__()
        self.coefficient = coefficient
        self.trainable = trainable

    def build(self, input_shape):
        """
        Ok
        :param input_shape:
        """

        iv = self.coefficient(shape=1,  dtype='float32')
        self.w = tf.Variable(initial_value= iv,
                             trainable=self.trainable)

    def call(self, inputs):
        return self.w * tf.math.abs(inputs[:, :, 1:2] - inputs[:, :, 0:1])

class TrivialLayer(layers.Layer):

    def __init__(self,dimension, dimension_before, num_pts,G_init = tf.random_uniform_initializer(0.0,1.0)):
        super(TrivialLayer, self).__init__()
        self.v = tf.Variable(initial_value = 0, trainable=False, dtype = 'float32')

    def call(self, input):
        return self.v

class PermutationMaxLayer(layers.Layer):

    def __init__(self,dimension, dimension_before, num_pts,G_init = tf.random_uniform_initializer(0.0,1.0)):
        super(PermutationMaxLayer, self).__init__()
        self.gamma = tf.Variable(initial_value = G_init(shape = (dimension_before,dimension), dtype = 'float32') , trainable=True)
        self.num_pts = num_pts
        self.dimension = dimension

    def call(self, inp):
        beta = tf.tile(tf.expand_dims(tf.reduce_max(inp, axis=1), 1), [1, self.num_pts, 1])
        return tf.reshape(tf.einsum("ijk,kl->ijl", beta, self.gamma), [-1, self.num_pts, self.dimension])

class PermutationSumLayer(layers.Layer):

    def __init__(self,dimension, dimension_before, num_pts,G_init = tf.random_uniform_initializer(0.0,1.0)):
        super(PermutationSumLayer, self).__init__()
        self.gamma = tf.Variable(initial_value = G_init(shape = (dimension_before,dimension), dtype = 'float32') , trainable=True)
        self.num_pts = num_pts
        self.dimension = dimension

    def call(self, inp):
        beta = tf.tile(tf.expand_dims(tf.reduce_sum(inp, axis=1), 1), [1, self.num_pts, 1])
        return tf.reshape(tf.einsum("ijk,kl->ijl", beta, self.gamma), [-1, self.num_pts, self.dimension])


class PeL(layers.Layer):

    def __init__(self, dimension, operationalLayer = TrivialLayer, bias_init = tf.random_uniform_initializer(0.0,1.0),
                 G_init = tf.random_uniform_initializer(0.0,1.0), L_init = tf.random_uniform_initializer(0.0,1.0)):
        super(PeL, self).__init__()
        self.dimension = dimension
        self.bias_init = bias_init
        self.operationalLayer = operationalLayer
        self.G_init = G_init
        self.L_init = L_init


    def build(self, input_shape):
        dimension_before, num_pts = input_shape[2], input_shape[1]
        self.b = tf.Variable(initial_value = self.bias_init(shape = (1,1,self.dimension), dtype = 'float32') , trainable=True)
        self.lbda = tf.Variable(initial_value = self.L_init(shape = (dimension_before, self.dimension), dtype = 'float32') , trainable=True)
        self.executableOpLayer = self.operationalLayer(self.dimension, dimension_before, num_pts, self.G_init)

    def call(self, inp):
        A = tf.reshape(tf.einsum("ijk,kl->ijl", inp, self.lbda), [-1, inp.shape[1], self.dimension])
        B = self.executableOpLayer(inp)
        return A - B + self.b

class SequencePeL(layers.Layer):
    def __init__(self, layerslist):
        super(SequencePeL, self).__init__()
        self.layerslist = layerslist

    def call(self, inp):
        output = inp
        for layer in self.layerslist:
            output = layer(output)

        return output



class BettiLayer(layers.Layer):
    """
This is BettiLayer
    """

    def __init__(self, theta, num_samples, sample_init=tf.random_uniform_initializer(0, 1.0)):
        super(BettiLayer, self).__init__()
        self.theta = theta
        self.num_samples = num_samples
        self.sample_init = sample_init

    def build(self, input_shape):
        """
        This is where we define the variables
        :param input_shape:
        """
        iv = self.sample_init(shape=[1, 1, self.num_samples], dtype='float32')
        self.sp = tf.Variable(iv, trainable=True)

    def call(self, inp):
        X, Y = inp[:, :, 0:1], inp[:, :, 1:2]
        return 1. / (1. + tf.exp(-self.theta * (.5 * (Y - X) - tf.abs(self.sp - .5 * (Y + X)))))

class LandscapeLayer(layers.Layer):

    def __init__(self,num_samples,sample_init = tf.random_uniform_initializer(0,1.0)):
        super(LandscapeLayer, self).__init__()
        self.num_samples = num_samples
        self.sample_init = sample_init

    def build(self, input_shape):
        """
        This is where we define the variables
        :param input_shape:
        """
        iv = self.sample_init(shape=[1, 1, self.num_samples], dtype='float32')
        self.sp = tf.Variable(iv, trainable=True)

    def call(self, inp):
        return tf.maximum(.5 * (inp[:, :, 1:2] - inp[:, :, 0:1]) - tf.abs(self.sp - .5 * (inp[:, :, 1:2] + inp[:, :, 0:1])), np.array([0]))

class ImageLayer(layers.Layer):
    def __init__(self, image_size, image_bnds = ((-0.0001, 1.0001), (-0.0001, 1.0001)), variance_init = tf.random_uniform_initializer(0,1.0)):
        super(ImageLayer, self).__init__()
        self.image_size = image_size
        self.image_bnds = image_bnds
        self.variance_init = variance_init

    def build(self, input_shape):
        iv = self.variance_init(shape=[1], dtype='float32')
        self.sg = tf.Variable(iv, trainable=True)

    def call(self, inp):
        bp_inp = tf.einsum("ijk,kl->ijl", inp, tf.constant(np.array([[1., -1.], [0., 1.]], dtype=np.float32)))
        dimension_before, num_pts = inp.shape[2], inp.shape[1]
        coords = [tf.range(start=self.image_bnds[i][0], limit=self.image_bnds[i][1],
                           delta=(self.image_bnds[i][1] - self.image_bnds[i][0]) / self.image_size[i]) for i in
                  range(dimension_before)]
        M = tf.meshgrid(*coords)
        mu = tf.concat([tf.expand_dims(tens, 0) for tens in M], axis=0)
        bc_inp = tf.reshape(bp_inp, [-1, num_pts, dimension_before]+ [1 for _ in range(dimension_before)])
        tmbp = tf.exp(tf.reduce_sum(-tf.square(bc_inp - mu) / (2 * tf.square(self.sg[0])), axis=2)) / (
                    2 * np.pi * tf.square(self.sg[0]))
        tmbp = tf.reshape(tmbp,[-1,num_pts,self.image_size[0] * self.image_size[1]])
        return tmbp


class ExponentialLayer(layers.Layer):
    def __init__(self, num_elements, coffee_init=tf.random_uniform_initializer(3.0, 3.0),
                 variance_init=tf.random_uniform_initializer(0, 1.0), variance_trainable=True):
        super(ExponentialLayer, self).__init__()
        self.num_elements = num_elements
        self.coffee_init = coffee_init
        self.variance_init = variance_init
        self.variance_trainable = variance_trainable

    def build(self, input_shape):
        db = input_shape[2]
        tinit = self.coffee_init(shape=(1, 1, db, self.num_elements),dtype='float32')
        self.t = tf.Variable(initial_value=tinit,trainable=True)
        sginit = self.variance_init(shape=(1, 1, db, self.num_elements),dtype='float32')
        self.sg = tf.Variable(initial_value=sginit, trainable=self.variance_trainable)

    def call(self, inp):
        bc_inp = tf.expand_dims(inp, -1)
        return tf.exp(tf.reduce_sum(-tf.multiply(tf.square(bc_inp - self.t), tf.square(self.sg)), axis=2))


class TopK(layers.Layer):
    def __init__(self, k):
        super(TopK, self).__init__()
        self.k = k

    def call(self, masked_layer):
        dimm = masked_layer.shape[2]
        masked_layer_t = tf.transpose(masked_layer, perm=[0, 2, 1])
        values, indices = tf.nn.top_k(masked_layer_t, self.k)

        return tf.reshape(values, [-1, self.k * dimm])

class SumLayer(layers.Layer):
    def __init__(self):
        super(SumLayer, self).__init__()

    def call(self, masked_layer):
        return tf.reduce_sum(masked_layer, axis=1)




#class MultiPerslayModel():

class BatchNormalizationLayer(layers.Layer):

    def __init__(self, layers):
        super(BatchNormalizationLayer).__init__()
        self.layers = layers
        self.alpha = tf.Variable(initial_value=tf.ones(shape = len(layers),dtype='float32'),trainable=True)

    def call(self, diag):

        list_layers = []

        for layer in self.layers:
            list_layers.append(layer(diag))

        list_dgm = [tf.multiply(self.alpha[idx], tf.layers.batch_normalization(dgm))
                    for idx, dgm in enumerate(list_layers)]

        return tf.math.add_n(list_dgm)


# class ImagePerslayModel(tf.keras.Model):
#     def __init__(self, rectangle_shape, num_label):
#         super(ImagePerslayModel, self).__init__()
#         self.rectangle_shape = rectangle_shape
#         self.ImageLayer = ImageLayer(rectangle_shape)
#         self.FlattenLayer = tf.keras.layers.Flatten()
#         self.DenseLayer = tf.keras.layers.Dense(num_label)
#         # activation='softmax'
#
#     def call(self, diag):
#         N, dimension_diag = diag.get_shape()[1], diag.get_shape()[2]
#         tensor_mask = diag[:, :, dimension_diag - 1]
#         tensor_diag = diag[:, :, :dimension_diag - 1]
#         tensor_diag = self.ImageLayer(tensor_diag)
#
#         tiled_mask = tf.tile(tf.expand_dims(tensor_mask, -1), [1, 1, tensor_diag.shape[2]])


class MultiPerslayModel(tf.keras.Model):
    def __init__(self, ModelList, num_labels):
        super(MultiPerslayModel, self).__init__()
        self.ModelList = ModelList
        self.DenseLayer = tf.keras.layers.Dense(num_labels)
        #activation='softmax'

    def call(self, diags):
        list_v = []
        for idx, diag_type in enumerate(diags):
            list_v.append(self.ModelList[idx](diag_type))
        representations = tf.concat(list_v, 1)
        final_output = self.DenseLayer(representations)
        return final_output

class SinglePerslayModel(tf.keras.Model):
    def __init__(self, CoefficentLayer, FunctionalLayer, OPLayer, num_labels):
        super(SinglePerslayModel, self).__init__()
        self.PerslayLayer = PerslayModel(CoefficentLayer, FunctionalLayer, OPLayer)
        self.DenseLayer = tf.keras.layers.Dense(num_labels)

    def call(self, diag):
        vector = self.PerslayLayer(diag)
        return self.DenseLayer(vector)

class PerslayModel(tf.keras.Model):
    def __init__(self, CoefficentLayer, FunctionalLayer, OPLayer):
        super(PerslayModel, self).__init__()
        self.CoefficentLayer = CoefficentLayer
        self.FunctionalLayer = FunctionalLayer
        self.OPLayer = OPLayer


    def call(self, diag):
        # First we split inputs into a mask and the actual variables
        N, dimension_diag = diag.get_shape()[1], diag.get_shape()[2]
        tensor_mask = diag[:, :, dimension_diag - 1]
        tensor_diag = diag[:, :, :dimension_diag - 1]

        # Next we compute the function and the coefficent
        weight = self.CoefficentLayer(tensor_diag)
        tensor_diag = self.FunctionalLayer(tensor_diag)

        # And now we combine coefficent with the function value
        tiled_weight = tf.tile(weight, [1, 1, tensor_diag.shape[2]])
        tensor_diag = tf.multiply(tensor_diag, tiled_weight)

        # Then we apply mask
        tiled_mask = tf.tile(tf.expand_dims(tensor_mask, -1), [1, 1, tensor_diag.shape[2]])
        masked_layer = tf.multiply(tensor_diag, tiled_mask)


        final_output = self.OPLayer(masked_layer)

        return final_output

    # print("hello")

### Exponential Layer

### Triangle layer

### TopK:

### Operational layer: Dense.

### Our model:
