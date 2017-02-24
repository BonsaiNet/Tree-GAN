"""Collection of functions to train the hierarchical model."""

from __future__ import print_function
import os
import pickle

import numpy as np

from keras.optimizers import RMSprop, Adagrad, Adam

import models as models
import batch_utils
import plot_utils
import visualization

import matplotlib.pyplot as plt



def clip_weights(model, weight_constraint):
    """
    Clip weights of a keras model to be bounded by given constraints.

    Parameters
    ----------
    model: keras model object
        model for which weights need to be clipped
    weight_constraint:

    Returns
    -------
    model: keras model object
        model with clipped weights
    """
    for l in model.layers:
        if True:  # 'dense' in l.name:
            weights = l.get_weights()
            weights = \
                [np.clip(w, weight_constraint[0],
                         weight_constraint[1]) for w in weights]
            l.set_weights(weights)
    return model


def save_model_weights(g_model, m_model, d_model,
                       level, epoch, batch, list_d_loss, model_path_root):
    """
    Save model weights.

    Parameters
    ----------
    g_model: keras model object
        geometry generator model
    m_model: keras model object
        morphology generator model
    d_model: keras model object
        discriminator model
    level: int
        level in the hierarchy
    epoch: int
        epoch #
    batch: int
        mini-batch #
    list_d_loss: list
        list of discriminator loss trace
    model_path_root: str
        path where model files should be saved
    """
    model_path = ('%s/level%s' % (model_path_root, level))

    g_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                          (g_model.name, epoch, batch))
    g_model.save_weights(g_file, overwrite=True)

    m_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                          (m_model.name, epoch, batch))
    m_model.save_weights(m_file, overwrite=True)

    d_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                          (d_model.name, epoch, batch))
    d_model.save_weights(d_file, overwrite=True)

    d_loss_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                               ('DiscLoss', epoch, batch))
    pickle.dump(list_d_loss, open(d_loss_file, "wb" ))


def train_model(training_data=None,
                n_nodes=20,
                input_dim=100,
                n_epochs=25,
                batch_size=32,
                n_batch_per_epoch=100,
                d_iters=20,
                lr_discriminator=0.005,
                lr_generator=0.00005,
                d_weight_constraint=[-.03, .03],
                g_weight_constraint=[-.03, .03],
                m_weight_constraint=[-.03, .03],
                rule='none',
                train_loss='wasserstein_loss',
                verbose=True):
    """
    Train the hierarchical model.

    Progressively generate trees with
    more and more nodes.

    Parameters
    ----------
    training_data: dict of dicts
        each inner dict is an array
        'geometry': 3-d arrays (locations)
            n_samples x n_nodes - 1 x 3
        'morphology': 2-d arrays
            n_samples x n_nodes - 1 (parent sequences)
        example: training_data['geometry']['n20'][0:10, :, :]
                 gives the geometry for the first 10 neurons
                 training_data['geometry']['n20'][0:10, :]
                 gives the parent sequences for the first 10 neurons
                 here, 'n20' indexes a key corresponding to
                 20-node downsampled neurons.
    n_nodes: array
        specifies the number of nodes.
    input_dim: int
        dimensionality of noise input
    n_epochs:
        number of epochs over training data
    batch_size:
        batch size
    n_batch_per_epoch: int
        number of batches per epoch
    d_iters: int
        number of iterations to train discriminator
    lr_discriminator: float
        learning rate for optimization of discriminator
    lr_generator: float
        learning rate for optimization of generator
    weight_constraint: array
        upper and lower bounds of weights (to clip)
    verbose: bool
        print relevant progress throughout training

    Returns
    -------
    geom_model: list of keras model objects
        geometry generators
    morph_model: list of keras model objects
        morphology generators
    disc_model: list of keras model objects
        discriminators
    gan_model: list of keras model objects
        discriminators stacked on generators
    """
    # ###################################
    # Initialize models
    # ###################################
    geom_model = list()
    morph_model = list()
    disc_model = list()
    gan_model = list()

    # Discriminator
    d_model = models.discriminator(n_nodes=n_nodes,
                                   batch_size=batch_size,
                                   train_loss=train_loss)
    # Generators and GANs
    g_model, m_model = \
        models.generator(n_nodes=n_nodes,
                         batch_size=batch_size)
    stacked_model = \
        models.discriminator_on_generators(g_model,
                                           m_model,
                                           d_model,
                                           conditioning_rule=rule,
                                           input_dim=input_dim,
                                           n_nodes=n_nodes)


    # Collect all models into a list
    disc_model.append(d_model)
    geom_model.append(g_model)
    morph_model.append(m_model)
    gan_model.append(stacked_model)

    # ###############
    # Optimizers
    # ###############
    optim_d = Adagrad()  # RMSprop(lr=lr_discriminator)
    optim_g = Adagrad()  # RMSprop(lr=lr_generator)

    # ##############
    # Train
    # ##############
    # ---------------
    # Compile models
    # ---------------

    g_model.compile(loss='mse', optimizer=optim_g)
    m_model.compile(loss='mse', optimizer=optim_g)

    d_model.trainable = False
    if train_loss == 'wasserstein_loss':
        stacked_model.compile(loss=models.wasserstein_loss,
                              optimizer=optim_g)
    else:
        stacked_model.compile(loss='binary_crossentropy',
                              optimizer=optim_g)

    d_model.trainable = True

    if train_loss == 'wasserstein_loss':
        d_model.compile(loss=models.wasserstein_loss,
                        optimizer=optim_d)
    else:
        d_model.compile(loss='binary_crossentropy',
                        optimizer=optim_d)

    if verbose:
        print("")
        print(20*"=")
    # -----------------
    # Loop over epochs
    # -----------------
    for e in range(n_epochs):
        batch_counter = 1
        g_iters = 0

        if verbose:
            print("")
            print("Epoch #{0}".format(e))
            print("")

        while batch_counter < n_batch_per_epoch:
            list_d_loss = list()
            list_g_loss = list()
            # ----------------------------
            # Step 1: Train discriminator
            # ----------------------------
            for d_iter in range(d_iters):

                # Clip discriminator weights
                d_model = clip_weights(d_model, d_weight_constraint)

                # Create a batch to feed the discriminator model
                select = range((batch_counter - 1) * batch_size ,
                               batch_counter * batch_size)
                X_locations_real = \
                    training_data['geometry']['n'+str(n_nodes)][select, :, :]
                X_locations_real = np.reshape(X_locations_real, [batch_size,
                                                                 (n_nodes - 1),
                                                                 3])
                X_parent_cut = \
                    np.reshape(training_data['morphology']['n'+str(n_nodes)][select, :],
                               [1, (n_nodes - 1) * batch_size])
                X_parent_real = \
                    batch_utils.get_batch(X_parent_cut=X_parent_cut,
                                          batch_size=batch_size,
                                          n_nodes=n_nodes)

                if train_loss == 'wasserstein_loss':
                    y_real = -np.ones((X_locations_real.shape[0], 1, 1))
                else:
                    y_real = np.ones((X_locations_real.shape[0], 1, 1))

                X_locations_gen, X_parent_gen = \
                    batch_utils.gen_batch(batch_size=batch_size,
                                           n_nodes=n_nodes,
                                           input_dim=input_dim,
                                           geom_model=g_model,
                                           morph_model=m_model,
                                           conditioning_rule=rule)

                if train_loss == 'wasserstein_loss':
                    y_gen = np.ones((X_locations_gen.shape[0], 1, 1))
                else:
                    y_gen = np.zeros((X_locations_gen.shape[0], 1, 1))
                # make data in half of real and generated
                cutting = int(batch_size/2)
                X_locations_real_first_half = np.append(X_locations_real[:cutting,:,:],
                                                        X_locations_gen[:cutting,:,:],
                                                        axis=0)
                X_parent_real_first_half = np.append(X_parent_real[:cutting,:,:],
                                                     X_parent_gen[:cutting,:,:],
                                                     axis=0)
                y_real_first_half = np.append(y_real[:cutting,:,:],
                                              y_gen[:cutting,:,:],
                                              axis=0)

                X_locations_real_second_half = np.append(X_locations_real[cutting:,:,:],
                                                         X_locations_gen[cutting:,:,:],
                                                         axis=0)
                X_parent_real_second_half = np.append(X_parent_real[cutting:,:,:],
                                                      X_parent_real[cutting:,:,:],
                                                      axis=0)
                y_real_second_half = np.append(y_real[cutting:,:,:],
                                               y_gen[cutting:,:,:],
                                               axis=0)
                # Update the discriminator
                disc_loss = \
                    d_model.train_on_batch([X_locations_real_first_half,
                                            X_parent_real_first_half],
                                            y_real_first_half)
                list_d_loss.append(disc_loss)
                disc_loss = \
                    d_model.train_on_batch([X_locations_real_second_half,
                                            X_parent_real_second_half],
                                            y_real_second_half)
                list_d_loss.append(disc_loss)

            if verbose:
                print("    After {0} iterations".format(d_iters))
                print("        Discriminator Loss \
                    = {0}".format(disc_loss))

            # -------------------------------------
            # Step 2: Train generators alternately
            # -------------------------------------
            # Freeze the discriminator
            d_model.trainable = False

            noise_input = np.random.rand(batch_size, 1, input_dim)

            gen_loss = \
                stacked_model.train_on_batch([noise_input],
                                             y_real)
            # Clip generator weights
            g_model = clip_weights(g_model, g_weight_constraint)
            m_model = clip_weights(m_model, m_weight_constraint)

            list_g_loss.append(gen_loss)
            if verbose:
                print("")
                print("    Generator_Loss: {0}".format(gen_loss))


            # Unfreeze the discriminator
            d_model.trainable = True

            # ---------------------
            # Step 3: Housekeeping
            # ---------------------
            g_iters += 1
            batch_counter += 1

            # Save model weights (few times per epoch)
            print(batch_counter)
            if batch_counter % 2 == 0:
                #m_model = clip_weights(m_model, m_weight_constraint)
                #g_model = clip_weights(g_model, g_weight_constraint)
                if verbose:
                    print ("     Level #{0} Epoch #{1} Batch #{2}".
                           format(1, e, batch_counter))

                    neuron_object = \
                        plot_utils.plot_example_neuron_from_parent(
                            X_locations_gen[0, :, :],
                            X_parent_gen[0, :, :])
                    plt.plot(np.squeeze(X_locations_gen[0, :, :]))

                    plot_utils.plot_adjacency(X_parent_real[0:2,:,:],
                                              X_parent_gen[0:2,:,:])

            # Display loss trace
            #if verbose:
                    plot_utils.plot_loss_trace(list_d_loss)

                    # save the models
                    save_model_weights(g_model,
                                       m_model,
                                       d_model,
                                       0,
                                       e,
                                       batch_counter,
                                       list_d_loss,
                                       model_path_root='../model_weights')

            #  Save models
            geom_model = g_model
            morph_model = m_model
            disc_model = d_model
            gan_model = stacked_model

    return geom_model, \
        morph_model, \
        disc_model, \
        gan_model
