"""Collection of Keras models for hierarchical GANs."""

# Imports
from keras.layers.core import Dense, Reshape, RepeatVector, Lambda, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

# Local imports
import layers as layers


# Generators
def generator(n_nodes=20,
              noise_dim=100,
              embedding_dim=100,
              hidden_dim=20,
              batch_size=64):
    """
    Generator network.
    Parameters
    ----------
    n_nodes: int
        number of nodes in the tree providing context input
    n_nodes: int
        number of nodes in the output tree
    noise_dim: int
        dimensionality of noise input
    embedding_dim: int
        dimensionality of embedding for context input
    Returns
    -------
    geometry_model: keras model object
        model of geometry generator
    conditional_geometry_model: keras model object
        model of geometry generator conditioned on morphology
    morphology_model: keras model object
        model of morphology generator
    conditional_morphology_model: keras model object
        model of morphology generator conditioned on geometry
    """
    # Generate noise input
    noise_input = Input(shape=(1, noise_dim), name='noise_input')

    # ---------------
    # Geometry model
    # ---------------

    # Dense
    #geometry_hidden_dim = (n_nodes - 1) * 20
    geometry_hidden1 = Dense(100)(noise_input)
    geometry_hidden2 = Dense(100)(geometry_hidden1)
    geometry_hidden3 = Dense(50)(geometry_hidden2)
    geometry_hidden4 = Dense(3 * (n_nodes - 1))(geometry_hidden3)
    #geometry_hidden3 = BatchNormalization()(geometry_hidden2)

    # Reshape
    geometry_reshaped = \
        Reshape(target_shape=(n_nodes - 1, 3))(geometry_hidden4)

    geometry_output = geometry_reshaped

    # Assign inputs and outputs of the model
    geometry_model = Model(input=[noise_input],
                           output=[geometry_output],
                           name="Geometry")

    # -----------------
    # Morphology model
    # -----------------

    # Dense
    #morphology_hidden_dim = n_nodes * 5
    morphology_hidden1 = Dense(100)(noise_input)
    morphology_hidden2 = Dense(100)(morphology_hidden1)
    # morphology_hidden2 = BatchNormalization()(morphology_hidden2)
    morphology_hidden3 = Dense(n_nodes * (n_nodes - 1),
                               activation='linear')(morphology_hidden2)

    # Reshape
    morphology_reshaped = \
        Reshape(target_shape=(n_nodes - 1, n_nodes))(morphology_hidden3)

    lambda_args = {'n_nodes': n_nodes, 'batch_size': batch_size}
    morphology_output = \
        Lambda(layers.masked_softmax,
               output_shape=(n_nodes - 1, n_nodes),
               arguments=lambda_args)(morphology_reshaped)

    # Assign inputs and outputs of the model

    morphology_model = \
        Model(input=[noise_input],
              output=[morphology_output],
              name="Morphology")

    geometry_model.summary()
    morphology_model.summary()
    return geometry_model, morphology_model


# Discriminator
def discriminator(n_nodes=20,
                  embedding_dim=100,
                  hidden_dim=50,
                  batch_size=64,
                  train_loss='wasserstein_loss'):
    """
    Discriminator network.
    Parameters
    ----------
    n_nodes: int
        number of nodes in the tree
    embedding_dim: int
        dimensionality of embedding for context input
    hidden_dim: int
        dimensionality of hidden layers
    Returns
    -------
    discriminator_model: keras model object
        model of discriminator
    """
    geometry_input = Input(shape=(n_nodes - 1, 3))
    morphology_input = Input(shape=(n_nodes - 1, n_nodes))

    # # Joint embedding of geometry and morphology
    # embedding = layers.embedder(geometry_input,
    #                             morphology_input,
    #                             n_nodes=n_nodes,
    #                             embedding_dim=embedding_dim)

    # Extract features from geometry and morphology
    lambda_args = {'n_nodes': n_nodes, 'batch_size': batch_size}
    n_features = 5 * n_nodes + 3
    both_inputs = merge([geometry_input,
                         morphology_input], mode='concat')
    embedding = \
        Lambda(layers.feature_extractor,
               output_shape=(n_nodes, n_features),
               arguments=lambda_args)([both_inputs])
    embedding = \
        Reshape(target_shape=(1, n_nodes * n_features))(embedding)

    # --------------------
    # Discriminator model
    # -------------------=
    discriminator_hidden1 = Dense(200)(embedding)
    # discriminator_hidden1 = Dropout(0.1)(discriminator_hidden1)
    discriminator_hidden2 = Dense(50)(discriminator_hidden1)
    # discriminator_hidden2 = Dropout(0.1)(discriminator_hidden2)
    discriminator_hidden3 = Dense(10)(discriminator_hidden2)
    # discriminator_hidden3 = Dropout(0.1)(discriminator_hidden3)

    if train_loss == 'wasserstein_loss':
        discriminator_output = \
            Dense(1, activation='linear')(discriminator_hidden3)
    else:
        discriminator_output = \
            Dense(1, activation='sigmoid')(discriminator_hidden3)

    discriminator_model = Model(input=[geometry_input,
                                       morphology_input],
                                output=[discriminator_output],
                                name="Discriminator")

    discriminator_model.summary()
    return discriminator_model


def wasserstein_loss(y_true, y_pred):
    """
    Custom loss function for Wasserstein critic.
    Parameters
    ----------
    y_true: keras tensor
        true labels: -1 for data and +1 for generated sample
    y_pred: keras tensor
        predicted EM score
    """
    return K.mean(y_true * y_pred)


# Discriminator on generators
def discriminator_on_generators(geometry_model,
                                morphology_model,
                                discriminator_model,
                                conditioning_rule='none',
                                input_dim=100,
                                n_nodes=20):
    """
    Discriminator stacked on the generators.
    Parameters
    ----------
    geometry_model: keras model object
        model object that generates the geometry
    conditional_geometry_model: keras model object
        model object that generates the geometry conditioned on morphology
    morphology_model: keras model object
        model object that generates the morphology
    conditional_morphology_model: keras model object
        model object that generates the morphology conditioned on geometry
    discriminator_model: keras model object
        model object for the discriminator
    conditioning_rule: str
        'mgd': P_w(disc_loss|g,m) P(g|m) P(m)
        'gmd': P_w(disc_loss|g,m) P(m|g) P(g)
    input_dim: int
        dimensionality of noise input
    n_nodes: int
        number of nodes in the tree providing
        prior context input for the generators
    n_nodes: int
        number of nodes in the output tree
    Returns
    -------
    model: keras model object
        model of the discriminator stacked on the generator.
    """
    # Inputs
    noise_input = Input(shape=(1, input_dim), name='noise_input')

    # ------------------
    # Generator outputs
    # ------------------
    if conditioning_rule == 'none':
        geometry_output = \
            geometry_model([noise_input])
        morphology_output = \
            morphology_model([noise_input])

    # ---------------------
    # Discriminator output
    # ---------------------
    discriminator_output = \
        discriminator_model([geometry_output,
                             morphology_output])

    # Stack discriminator on generator
    model = Model([noise_input],
                  [discriminator_output])

    return model
