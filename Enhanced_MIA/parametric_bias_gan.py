import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Lambda
import sys
from Adult_utils import *
from fairness_utils import *
from Adult_autoencoder import Autoencoder
import seaborn as sns



def plot_gan_convergence(d1, d2, gen_d1, gen_d2) :
    epochs = list(range(1, len(d1) + 1))
    data = {
        'Epochs': epochs,
        'Discriminator 1': d1,
        'Discriminator 2': d2,
        'Generator w.r.t Discriminator 1': gen_d1,
        'Generator w.r.t Discriminator 2': gen_d2
        }
    df = pd.DataFrame(data)

    melted_df = pd.melt(df, id_vars='Epochs', value_vars=['Discriminator 1', 'Discriminator 2',
                                                      'Generator w.r.t Discriminator 1',
                                                      'Generator w.r.t Discriminator 2'],
                    var_name='Loss Type', value_name='Loss')

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    plot = sns.lineplot(x='Epochs', y='Loss', hue='Loss Type', data=melted_df)

    plot.set(xlabel='Epochs', ylabel='Loss', title='Convergence Curve')
    plt.show()

def dataSet_summary2(data) :
    print(data.size, "entry in this Dataset")
    print(data.head())
    print(data.shape)
    print ("Null values check : ")
    print ("\n", data.isnull().sum())
    print ("\n \nBalance check : ")

    class_1 = data[data['income'] == 1.0]
    class_0 = data[data['income'] == 0.0]
    ratio = min(class_0.size / class_1.size, class_1.size / class_0.size)
    print("label balance level : ", round(ratio, 5) * 100, "%")
    rate = 100 * round(data[data['sex'] == 1.0].size / data.size, 2)
    print("Male rate is : "+ str(rate))
    rate = 100 * round(data[data['sex'] == 0.0].size / data.size, 2)
    print("Female rate is : "+ str(rate))
    rate = 100 * round(data[data['Black'] == 1.0].size / data.size, 2)


"""
    Customized loss_function : Add a penalty to binary_cross_entropy in order to reduce
    discriminator2 performances
"""
def custom_discriminator_loss(y_true, y_pred):
    epsilon = 0.9 #penalty term:  epsilone = 0 -> FAIRGAN
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # Or any other loss function you are using
    # A regularization term to penalize loss going below epsilon
    loss += tf.math.maximum(0.0, epsilon - loss)  # Penalize if loss < epsilon
    return loss

def gen_loss(fake_data) :
    return binary_cross_entropy(fake_data)

class GAN:
    def __init__(self, latent_dim, salient_dim, data_dim, _lambda_) :
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.salient_dim = salient_dim
        self.discriminator1, self.discriminator2 = self.build_discriminator()
        self.generator = self.build_generator()
        intermediate_layer = (self.data_dim + self.salient_dim)/2
        self.compress_dims = [self.data_dim, intermediate_layer, self.salient_dim]  # Two compression layers to dimensions 64
        self.decompress_dims = [self.salient_dim, intermediate_layer, self.data_dim]           # Decompress dimensions salient Representation has dimension 64
        self.autoencoder, self.encoder_model, self.decoder_model = self.build_autoencoder(self.compress_dims, self.decompress_dims)
        self.gan = self.build_gan(_lambda_)


    def build_autoencoder(self, compress_dim, decompress_dims) :

        model = Autoencoder(self.compress_dims, self.decompress_dims, data_type='float64', l2_scale=0)
        encoder_layers = model.layers[:len(self.compress_dims)]
        # Encoder contains only the encoder layers
        encoder_model = tf.keras.Sequential(encoder_layers)
        decoder_layers = model.layers[len(self.compress_dims):]  # Get the decoder layers
        # Create a new model containing only the decoder layers
        decoder_model = tf.keras.Sequential(decoder_layers)
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model, encoder_model, decoder_model


    def build_discriminator(self):
        discriminator1 = Sequential()
        discriminator1.add(Dense(256, input_dim=self.data_dim, activation='relu'))
        discriminator1.add(Dense(128, activation='relu'))
        discriminator1.add(Dense(16, activation='relu'))
        discriminator1.add(Dense(1, activation='sigmoid')) #binary classification -> sigmoid
        discriminator1.compile(loss=tf.keras.metrics.binary_crossentropy,
                                optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001))

        discriminator2 = Sequential()
        discriminator2.add(Dense(16, input_dim=self.data_dim-1, activation='relu'))
        discriminator2.add(Dense(128, activation='relu'))
        discriminator2.add(Dense(16, activation='relu'))
        discriminator2.add(Dense(1, activation='sigmoid'))  #binary classification -> sigmoid
        discriminator2.compile(loss=tf.keras.metrics.binary_crossentropy, optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001))  # Loss is binary cross entropy

        return discriminator1, discriminator2

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', kernel_initializer = tf.initializers.HeUniform(seed=SEED), bias_initializer = tf.initializers.HeUniform(seed=SEED), input_dim=self.latent_dim))
        #model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(128, activation='relu', kernel_initializer = tf.initializers.HeUniform(seed=SEED), bias_initializer = tf.initializers.HeUniform(seed=SEED)))
        model.add(Dense(100, activation='relu', kernel_initializer = tf.initializers.HeUniform(seed=SEED), bias_initializer = tf.initializers.HeUniform(seed=SEED)))
        #After preprocessing : all the values in the dataset are in the [0, 1] interval so Sigmoid is used in the output layer of the decoder
        model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
        model.add(Dense(self.salient_dim, activation='tanh'))
        model.compile(loss=tf.keras.metrics.binary_crossentropy, optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01))

        return model



    def train_step(self, mode, x, y, direction='DESCENDING', lr=0.0001, some_data=None) :

        with tf.GradientTape() as tape :
            if mode == 'disc1' :
                #Forward pass
                y_pred_real = self.discriminator1(x[:64], training=True)
                y_pred_fake = self.discriminator1(x[64:], training=True)
                batch_size = 64
                real_loss = self.discriminator1.loss(y_pred_real, y[:batch_size])
                fake_loss = self.discriminator1.loss(y_pred_fake, y[batch_size:])
                loss = real_loss + fake_loss
                gradients = tape.gradient(tf.reduce_mean(loss), self.discriminator1.trainable_variables)
                #Differentiate and backpropagate
                for param, grad in zip(self.discriminator1.trainable_variables, gradients):
                    if direction == 'ASCENDING' :
                        param.assign_add(grad * lr)
                    if direction == 'DESCENDING' :
                        param.assign_add(grad * (-lr))
                return tf.reduce_mean(loss)

            if mode == 'disc2' :
                y_pred = self.discriminator2(x, training=True)
                loss = self.discriminator2.loss(y_pred, y)
                gradients = tape.gradient(tf.reduce_mean(loss), self.discriminator2.trainable_variables,
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
                for param, grad in zip(self.discriminator2.trainable_variables, gradients):
                    if direction == 'ASCENDING' :
                        param.assign_add(grad * lr)
                    if direction == 'DESCENDING' :
                        param.assign_add(grad * (-lr))
                return tf.reduce_mean(loss)

            #Generator training w.r.t D1
            if mode == 'gen_d1' :   #Train gen to fool d_1 -> x
                self.generator.trainable = True
                #Evaluate the fake data using D_1
                noise = np.random.normal(0, 1, (64, self.latent_dim))
                d1_pred = self.discriminator1(self.generator(noise, training=True))
                d1_labels = tf.reshape(tf.ones(d1_pred.shape[0], dtype='float64'), d1_pred.shape)
                tape.watch(self.generator.trainable_variables)
                loss = self.generator.loss(d1_labels, self.discriminator1(self.generator(x, training=False)))
                gradients = tape.gradient(loss, self.generator.trainable_variables,
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
                if None in gradients :
                    print('None in gradients : ', gradients)
                    exit()
                for param, grad in zip(self.generator.trainable_variables, gradients):
                    if direction == 'ASCENDING' :
                        param.assign_add(grad * lr)
                    if direction == 'DESCENDING' :
                        param.assign_add(grad * (-lr))
                return tf.reduce_mean(loss)

            #Generator training w.r.t both discriminators
            if mode == 'gen_d1_d2' :
                self.generator.trainable = True
                #Evaluate the fake data using D_1
                fake_data = self.generator(x, training=True)
                d1_pred = self.discriminator1(fake_data)
                d1_labels = tf.reshape(tf.ones(d1_pred.shape[0], dtype='float64'), d1_pred.shape)
                loss_d1 = self.generator.loss(d1_labels, self.discriminator1(self.generator(x, training=False)))

                #Remove the sensitive attribute
                fake_data_new = tf.concat([fake_data[:, :3], fake_data[:, 3 + 1:]], axis=1)
                d2_pred = self.discriminator2(fake_data_new)
                 #Fool d1 ---> Reverse labels and compute associated loss
                d2_labels = np.where(fake_data[:, 3]>=0.5, 0.0, 1.0)
                loss_d2 = self.generator.loss(d2_labels.reshape(-1, 1), d2_pred)
                loss_d2 = tf.reduce_mean(loss_d2)
                loss = loss_d1 + loss_d2
                tape.watch(self.generator.trainable_variables)
                gradients = tape.gradient(loss, self.generator.trainable_variables,
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
                if None in gradients :
                    print('None in gradients : ', gradients)
                    exit()
                for param, grad in zip(self.generator.trainable_variables, gradients):
                    if direction == 'ASCENDING' :
                        param.assign_add(grad * lr)
                    if direction == 'DESCENDING' :
                        param.assign_add(grad * (-lr))
                return (tf.reduce_mean(loss_d1), tf.reduce_mean(loss_d2))


    def build_gan(self, _lambda_):
        gan_input = Input(shape=(self.latent_dim,))
        gan_output = self.build_generator()(gan_input)
        _, _, self.decoder_model = self.build_autoencoder(self.compress_dims, self.decompress_dims)
        generated_data = self.decoder_model(gan_output)
        validity1 = self.discriminator1(generated_data)
        generated_data_without_attribute = Lambda(lambda x: tf.concat([x[:, :3], x[:, 3+1:]], axis=1))(generated_data)
        validity2 = self.discriminator2(generated_data_without_attribute)
        _lambda_ = 1.0
        combined_loss = ['binary_crossentropy', 'binary_crossentropy']  # Loss for discriminator1 and discriminator2
        combined_loss_weights = [1.0, _lambda_]     # Weights for combining losses : w1L1 + w2L2

        combined_model = Model(gan_input, [validity1, validity2])
        combined_model.compile(loss=combined_loss,
                               loss_weights=combined_loss_weights,
                               optimizer='adam')

        return combined_model


    def train(self, data, epochs, autoencoder_epochs, autoencoder_batch_size, GAN_batch_size):
        sensitive_pos = 3 #Gender attribute position
        print('pre_training the autoencoder model ...')
        self.autoencoder.fit(data, data, epochs=autoencoder_epochs, batch_size=autoencoder_batch_size)
        # Point to the encoder and decoder
        decoder_layers = self.autoencoder.layers[len(self.compress_dims):]  # Get the decoder layers
        # Get the trained decoder layers
        decoder_model = tf.keras.Sequential(decoder_layers)
        #Decode(Gen(z))
        self.generator.add(decoder_model)
        #Store losses for plotting
        d1_loss = []
        d2_loss = []
        gen_d1_loss_array = []
        gen_d2_loss_array = []
        #Recompile the full generator
        self.generator.compile(loss=tf.keras.metrics.binary_crossentropy, optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001))
        for epoch in range(epochs):
            for i in range(round(data.shape[0] // GAN_batch_size)):
                # Train discriminator1
                self.discriminator1.trainable = True
                #Select a batch of data
                real_data = data[np.random.randint(0, data.shape[0], GAN_batch_size)]
                #Generate a batch of data
                noise = np.random.normal(0, 1, (GAN_batch_size, self.latent_dim))
                fake_data = self.generator.predict(noise)
                x = np.vstack([np.round(fake_data), real_data])
                y = np.vstack([np.zeros(GAN_batch_size), np.ones(GAN_batch_size)]).reshape(-1, 1)
                #train d1
                disc1_loss = self.train_step('disc1', x, y, 'DESCENDING')

                disc1_labels = np.ones((GAN_batch_size, 1))
                gen_d1_loss = self.train_step('gen_d1', noise, disc1_labels, 'DESCENDING', some_data=real_data)
                disc2_loss = tf.Variable(0.0)
                gen_d2_loss = tf.Variable(0.0)
                if epoch >= 25 : #start D2 at 500
                    #train d2
                    disc2_labels = np.where(fake_data[:, sensitive_pos] >=0.5, 1.0, 0.0)
                    fake_data_new = np.delete(fake_data, sensitive_pos, axis=1)
                    self.discriminator2.trainable = True
                    disc2_loss = self.train_step('disc2', fake_data_new, disc2_labels.reshape(-1, 1), 'DESCENDING')
                    #Fooling discriminator1 : Disc_1 predicts all synthetic data as real : y = 1
                    # Fooling discriminator2 : predict 0 when s = 1 and 1 when s = 0
                    (gen_d1_loss, gen_d2_loss) = self.train_step('gen_d1_d2', noise, disc2_labels, 'DESCENDING', some_data=real_data)

                    #print losses from time to time
                if i % 100 == 0 :
                    print(f"\nbatch {i} epoch {epoch}")
                    print('disc1 loss = ', disc1_loss)
                    print('disc2 loss = ', disc2_loss)
                    print('gen_d1_loss = ', gen_d1_loss)
                    print('gen_d2_loss = ', gen_d2_loss)
                #combined_loss = self.gan.train_on_batch(noise, combined_labels)
            print(f"Epoch {epoch}, D1 Loss: {disc1_loss}, D2 Loss: {disc2_loss}, G Loss: [D_1 : {gen_d1_loss}, D2 : {gen_d2_loss}]")
            d1_loss.append(disc1_loss.numpy())
            d2_loss.append(disc2_loss.numpy())
            gen_d1_loss_array.append(gen_d1_loss.numpy())
            gen_d2_loss_array.append(gen_d2_loss.numpy())


            # Open the file in write mode and write loss values
        with open('D1_convergence.txt', 'w') as f_d1, open('D2_convergence.txt', 'w') as f_d2, open('gen_D1_convergence.txt', 'w') as f_gen_d1, open('Gen_D2_convergence.txt', 'w') as f_gen_d2 :

            for (d1_val, d2_val, gen_d1_val, gen_d2_val) in zip(d1_loss, d2_loss, gen_d1_loss_array, gen_d2_loss_array) :
                f_d1.write(f'{d1_val}\n')
                f_d2.write(f'{d2_val}\n')
                f_gen_d1.write(f'{gen_d1_val}\n')
                f_gen_d2.write(f'{gen_d2_val}\n')


    def generate_samples(self, num_samples, cols):
        num_samples = 20000
        #sample num_samples of noise
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))

        # Use the trained generator to generate synthetic data
        generated_data = self.generator.predict(noise)
        generated_dataframe = pd.DataFrame(generated_data, columns=cols)
        return generated_dataframe



def main(_lambda_):

    latent_dim = 100  #latent space
    salient_dim = 75 # < data_dim
    data = pd.read_csv('datasets/adult.csv')
    data.replace('?', np.NaN)

    GAN_batch_size = 64
    autoencoder_batch_size = 128
    processed_data = data_PreProcess(data)
    data_dim = processed_data.shape[1]  # 89
    gan = GAN(latent_dim, salient_dim, data_dim, _lambda_)

    GAN_epochs = 20   #GAN epochs 2000
    autoencoder_epochs = 200  #200
    gan.train(processed_data.to_numpy(), GAN_epochs, autoencoder_epochs, autoencoder_batch_size, GAN_batch_size)

    num_samples = 100#10k samples dataset
    generated_dataframe = gan.generate_samples(num_samples, processed_data.columns)
    #dataSet_summary2(generated_dataframe)
    #file_path = 'datasets/new_lambda_manipulation/'+str(_lambda_)+'-UNFAIRGAN_fake_adult.csv'

    # Write the synthetic DataFrame to a CSV file
    #generated_dataframe.to_csv(file_path, index=False)

if __name__ == "__main__":
    _lambda_ = float(sys.argv[1])
    #main(_lambda_)

    latent_dim = 100  #latent space
    salient_dim = 70 # < data_dim
    data = pd.read_csv('datasets/adult.csv')
    data.replace('?', np.NaN)

    GAN_batch_size = 64
    autoencoder_batch_size = 128
    processed_data = data_PreProcess(data)
    data_dim = processed_data.shape[1]  # 89
    gan = GAN(latent_dim, salient_dim, data_dim, _lambda_)

    GAN_epochs = 50             #GAN epochs 2000
    autoencoder_epochs = 1000   #200
    gan.train(processed_data.to_numpy(), GAN_epochs, autoencoder_epochs, autoencoder_batch_size, GAN_batch_size)

    num_samples = 100#10k samples dataset
    generated_dataframe = gan.generate_samples(num_samples, processed_data.columns)

    file_path = 'GAN_training_loss/training10/'
    #Compare DT and DI of synthetic data with original dataset's
    d1_file_path = file_path+'D1_convergence.txt'
    d2_file_path = file_path+'D2_convergence.txt'
    gen_d1_file_path = file_path+'gen_D1_convergence.txt'
    gen_d2_file_path = file_path+'Gen_D2_convergence.txt'

    # List to store the floats
    d1_list = []
    d2_list = []
    gen_d1_list = []
    gen_d2_list = []
    # Open the file in read mode ('r')
    with open(d1_file_path, 'r') as file:
        # Read each line from the file
        for line in file:
            # Convert the line to a float and append to the list
            float_value = float(line.strip())  # strip() removes newline characters
            d1_list.append(float_value)
    with open(d2_file_path, 'r') as file:
        # Read each line from the file
        for line in file:
            # Convert the line to a float and append to the list
            float_value = float(line.strip())  # strip() removes newline characters
            d2_list.append(float_value)

    with open(gen_d1_file_path, 'r') as file:
        # Read each line from the file
        for line in file:
            # Convert the line to a float and append to the list
            float_value = float(line.strip())  # strip() removes newline characters
            gen_d1_list.append(float_value)

    with open(gen_d2_file_path, 'r') as file:
        # Read each line from the file
        for line in file:
            # Convert the line to a float and append to the list
            float_value = float(line.strip())  # strip() removes newline characters
            gen_d2_list.append(float_value)

    plot_gan_convergence(d1_list, d2_list, gen_d1_list, gen_d2_list)
