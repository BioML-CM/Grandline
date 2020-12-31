import tensorflow as tf
import numpy as np
import scipy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool1D, AveragePooling1D, Activation
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers
import tensorflow.keras.backend as K

from . import graph

class ChebyshevLayer(tf.keras.layers.Layer):

    def __init__(self, L, Fin, Fout, K, regularize_rate, seed):
        super(ChebyshevLayer, self).__init__()
        self.Fin = int(Fin)
        self.Fout = int(Fout)
        self.K = int(K)
        self.regularize_rate = regularize_rate
        self.seed = seed

        self.L = scipy.sparse.csr_matrix(graph.rescale_L(L, lmax=2), dtype=np.float32).tocoo() # sparse matrix

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Fin': self.Fin,
            'Fout': self.Fout,
            'K': self.K,
            'regularize_rate': self.regularize_rate,
            'seed': self.seed,
            'L': self.L,
        })

        return config

    def build(self, input_shape):
        self.N, self.M, self.Fin = input_shape
        self.W = self.add_weight(shape=(self.Fin, self.K, self.Fout), 
                                 initializer=initializers.glorot_normal(seed=self.seed), 
                                 trainable=True, 
                                 name='W')
        self.b = self.add_weight(shape=[1, 1, self.Fout], initializer='zeros', trainable=True, name='b')


    def call(self, inputs):

        self.add_loss(self.regularize_rate * tf.nn.l2_loss(self.W))

        N, M, Fin = self.N, self.M, self.Fin

        indices = np.column_stack((self.L.row, self.L.col))
        L = tf.sparse.SparseTensor(indices=indices, values=self.L.data, dense_shape=self.L.shape)
        L = tf.sparse.reorder(L)

        X0 = tf.transpose(inputs, perm=[1, 2, 0]) # M, Fin, N
        X0 = tf.reshape(X0, [M, -1]) # M x Fin*N
        X = tf.expand_dims(X0, 0) # 1, M, Fin*N

        def concat(X, X_):
            X_ = tf.expand_dims(X_, 0)
            return tf.concat([X, X_], axis=0)

        if self.K > 1:
            X1 = tf.sparse.sparse_dense_matmul(L, X0)
            X = concat(X, X1)
        for k in range(2, self.K):
            X2 = 2 * tf.sparse.sparse_dense_matmul(L, X1) - X0 # M x Fin*N
            X = concat(X, X2)
            X0, X1 = X1, X2

        X = tf.reshape(X, [self.K, M, Fin, -1])  # K x M x Fin x N
        X = tf.transpose(X, perm=[3,1,2,0])  # N x M x Fin x K


        X = tf.einsum('aijk,jkl->ail', X, self.W)
        X = X + self.b


        return X
        

def build_gcn_model(As, Ls, Fs, Ks, Ps, Ms,
                    learning_rate=0.001,
                    regularization=0,
                    num_epochs=100, 
                    batch_size=50, 
                    eval_frequency=10,
                    filter_name='chebyshev',
                    activation='relu',
                    dropout=None,
                    decay_rate=None,
                    momentum=None,
                    seed=42,
                    dir_name=None
                   ):
    
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    ##### Create a model #####
    
    num_nodes = Ls[0].shape[0]
    inputs = tf.keras.Input(shape=(num_nodes, 1))
        
    # Add GCN layers

    for i, (Fout, K, P) in enumerate(zip(Fs, Ks, Ps)):
        if i == 0:
            Fin = 1
            x = ChebyshevLayer(Ls[0], Fin, Fout, K, regularization, seed)(inputs)
        else:
            L_idx = int(np.sum(np.log2(Ps[:i])))
            Fin = Fs[i-1]
            x = ChebyshevLayer(Ls[L_idx], Fin, Fout, K, regularization, seed)(x)
        x = Activation(activation)(x)
        x = AveragePooling1D(pool_size=P, strides=P, padding='same')(x)

    
    # Flaten and FC layers
    x = Flatten()(x)
    
    for l in Ms[:-1]:
        x = Dense(l, activation=activation, activity_regularizer=l2(regularization), kernel_initializer=initializers.glorot_normal(seed=seed))(x)
        
    # Last logit layer
    dense = Dense(Ms[-1], kernel_initializer=initializers.glorot_normal(seed=seed))(x)
    outputs = Activation(activations.softmax)(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model_logit = tf.keras.Model(inputs=inputs, outputs=dense)
    
    return model, model_logit



def cal_gradcam(selected_sample_id, X_input, model):
    
    for i, l in enumerate(model.layers):
        if 'chebyshev' in l.name:
            last_conv_layer = model.layers[i+1]
    
    grad_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(X_input)
        class_idx = np.argmax(predictions)
        loss = predictions[:, class_idx]

    last_conv_layer_output = conv_outputs # [selected_sample_id]
    grads = tape.gradient(loss, conv_outputs) # [selected_sample_id]

    
    alpha = grads.numpy().sum(axis=0) # or mean
    feature_map = last_conv_layer_output.numpy()
    
    node_importance = (feature_map * alpha).sum(axis=-1) # or mean
    
    return node_importance.flatten()
