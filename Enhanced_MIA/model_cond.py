import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Concatenate, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model

def generator_model():
    latent_dim = 100  # Dimension of the latent space
    label_input = Input(shape=(1,))  # Binary label (0 or 1)
    noise_input = Input(shape=(latent_dim,))  # Noise vector of dimension 100
    # Concatenate noise vector and binary label
    concatenated_input = Concatenate(axis=1)([noise_input, label_input])
    # Generator layers
    x = Dense(128)(concatenated_input)
    x = BatchNormalization()(x)  # Adding Batch Normalization for better training stability
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(88, activation='sigmoid')(x)  # Output a vector of size 88

    # Create the conditional generator model
    conditional_generator = Model(inputs=[noise_input, label_input], outputs=output)
    return conditional_generator

def discriminator_model():
    # Define input layers
    feature_input = Input(shape=(88,))  # Feature vector of size 88
    label_input = Input(shape=(1,))     # Binary label (0 or 1)
    # Concatenate feature vector and binary label
    concatenated_input = Concatenate(axis=1)([feature_input, label_input])
    # Discriminator layers
    x = Dense(128)(concatenated_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(1, activation='sigmoid')(x)  # Output a probability score (0 to 1)

    # Create the conditional discriminator model
    conditional_discriminator = Model(inputs=[feature_input, label_input], outputs=output)

    return conditional_discriminator

def fairness_discriminator_model():
    # Define input layers
    feature_input = Input(shape=(88,))  # Feature vector of size 88
    # Discriminator layers
    x = Dense(128)(feature_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Dense(1, activation='sigmoid')(x)  # Output a probability score (0 to 1)

    # Create the conditional discriminator model
    fairness_discriminator = Model(inputs=feature_input, outputs=output)

    return fairness_discriminator
