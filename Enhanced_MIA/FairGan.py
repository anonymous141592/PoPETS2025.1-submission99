import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models
import time
import cv2
import model_cond
import pandas as pd
from Adult_utils import *
import seaborn as sns


batch_size=256
epochs=2600
start_d2=1000
noise_dim=100
_lambda_ = float(sys.argv[1])
num_examples_to_generate=25000
checkpoint_dir='./training_checkpoints_gan_cond'
checkpoint_prefix=os.path.join(checkpoint_dir, "ckpt")
d1_loss = []
d2_loss = []
g_loss = []
g_loss2 = []

model_directory_path = './training_checkpoints_gan_cond'
data_directory_path = 'new_tests5/checkpoint_samples_gan_eps='+str(_lambda_)

if not os.path.exists(model_directory_path):
    os.makedirs(model_directory_path)
if not os.path.exists(data_directory_path):
    os.makedirs(data_directory_path)

data = pd.read_csv('../datasets/adult.csv')
data.replace('?', np.NaN)

train_dataset = data_PreProcess(data)

def plot_gan_convergence(d1, d2, gen) :
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), d1, marker='o', label='D1 Loss')
    plt.plot(range(1, epochs + 1), d2, marker='o', label='D2 Loss')
    plt.plot(range(1, epochs + 1), gen, marker='o', label='Generator Loss')

    plt.title('Convergence of the Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.show()


def discriminator1_loss(real_output, fake_output):
    real_loss=cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss=cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss=real_loss+fake_loss
    return total_loss

def discriminator2_loss(predited_labels, real_labels):
    real_labels = tf.cast(real_labels, dtype=tf.float32)
    loss=cross_entropy(predited_labels, tf.reshape(real_labels, (-1, 1)))
    #loss += tf.math.maximum(0.0, epsilon - loss)  # Penalize if loss < epsilon
    return loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_loss2(predicted_sens_attr):
    new_lables = tf.ones_like(predicted_sens_attr) * 0.5
    return cross_entropy(predicted_sens_attr, new_lables)

def generator_loss3(predicted_sens_attr):
    return -tf.reduce_mean(tf.math.log(predicted_sens_attr + 1e-8))
#Create different models
generator=model_cond.generator_model()
discriminator=model_cond.discriminator_model()
discriminator2=model_cond.fairness_discriminator_model()
#loss function
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer=tf.keras.optimizers.Adam(1E-4)
discriminator_optimizer=tf.keras.optimizers.Adam(1E-4)
discriminator2_optimizer=tf.keras.optimizers.Adam(1E-4)

checkpoint=tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

seed=tf.random.normal([num_examples_to_generate, noise_dim])

#Training without D2
def train_step_v1(real_data_batch):
    noise=tf.random.normal([real_data_batch.shape[0], noise_dim])
    generated_sens_attr=tf.random.uniform(shape=[real_data_batch.shape[0]], minval=0, maxval=2, dtype=tf.dtypes.int32)
    #generated_labels=tf.one_hot(generated_labels, 10)
    real_sens_attr = real_data_batch[:, 3]
    real_data_batch = np.delete(real_data_batch, 3, axis=1)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape :
      generated_data=generator([noise, generated_sens_attr], training=True)

      real_output=discriminator([real_data_batch, real_sens_attr], training=True)
      fake_output=discriminator([generated_data, generated_sens_attr], training=True)

      gen_loss=generator_loss(fake_output)
      disc_loss=discriminator1_loss(real_output, fake_output)

    gradients_of_generator=gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator=disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return (gen_loss, disc_loss)

#trainng with D2
def train_step(real_data_batch):
    noise=tf.random.normal([real_data_batch.shape[0], noise_dim])
    generated_sens_attr=tf.random.uniform(shape=[real_data_batch.shape[0]], minval=0, maxval=2, dtype=tf.dtypes.int32)
    #generated_labels=tf.one_hot(generated_labels, 10)
    real_sens_attr = real_data_batch[:, 3]
    real_data_batch = np.delete(real_data_batch, 3, axis=1)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as disc2_tape:
      generated_data=generator([noise, generated_sens_attr], training=True)

      real_output=discriminator([real_data_batch, real_sens_attr], training=True)
      fake_output=discriminator([generated_data, generated_sens_attr], training=True)
      predicted_sens_attr = discriminator2(generated_data, training=True)

      gen_loss=generator_loss(fake_output)
      disc_loss=discriminator1_loss(real_output, fake_output)
      disc2_loss=discriminator2_loss(predicted_sens_attr, generated_sens_attr)
      gen_loss2=generator_loss2(predicted_sens_attr)
      gen_loss += _lambda_ * gen_loss2

    gradients_of_generator=gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator=disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_discriminator2=disc2_tape.gradient(disc2_loss, discriminator2.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    discriminator2_optimizer.apply_gradients(zip(gradients_of_discriminator2, discriminator2.trainable_variables))
    return (gen_loss, gen_loss2, disc_loss, disc2_loss)

def train(dataset, epochs):
  for epoch in range(epochs):
    start=time.time()

    for i in range(round(dataset.shape[0] // batch_size)):
      #Select a batch of data
      real_data_batch = dataset.sample(n=batch_size)
      if epoch >= start_d2 :
          gen_loss, gen_loss2, disc_loss, disc2_loss = train_step(real_data_batch.to_numpy(dtype='float32'))
          d1_loss.append(disc_loss.numpy())
          d2_loss.append(disc2_loss.numpy())
          g_loss.append(gen_loss.numpy())
          g_loss2.append(gen_loss2.numpy())
      else :
          gen_loss, disc_loss = train_step_v1(real_data_batch.to_numpy(dtype='float32'))
          d1_loss.append(disc_loss.numpy())
          d2_loss.append(0.0)
          g_loss.append(gen_loss.numpy())
          g_loss2.append(0.0)


    print('gen loss w.r.t D1: ', g_loss[-1])
    print('gen loss w.r.t D2: ', g_loss2[-1])
    print('disc loss : ', d1_loss[-1])
    print('disc2 loss : ', d2_loss[-1])
    #save every 100 epochs
    if (epoch+1)%200==0:
      generate_and_save_samples(generator, epoch+1, seed, dataset.columns)
      checkpoint.save(file_prefix=checkpoint_prefix)
    print ('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))


def generate_and_save_samples(model, epoch, test_input, cols):
  labels=tf.random.uniform(shape=[num_examples_to_generate], minval=0, maxval=2, dtype=tf.dtypes.int32)
  generated_data=model([test_input, labels], training=False)
  print(generated_data)
  generated_data = np.insert(generated_data, 3, labels, axis=1)
  generated_data = pd.DataFrame(generated_data, columns=cols)
  generated_data.to_csv(data_directory_path+'/epoch='+str(epoch)+'.csv', index=False)


train(train_dataset, epochs)
#plot_gan_convergence(d1_loss, d2_loss, g_loss)
