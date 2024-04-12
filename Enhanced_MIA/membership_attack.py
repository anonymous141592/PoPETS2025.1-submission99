import numpy as np
import pandas as pd
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import sklearn
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
#from scikeras.wrappers import KerasClassifier
from art.estimators.classification import KerasClassifier

from Adult_utils import *
from fairness_utils import *

from sklearn.linear_model import LinearRegression
from art.estimators.regression.scikitlearn import ScikitlearnRegressor
from art.estimators.regression.keras import KerasRegressor
from art.utils import load_diabetes
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.membership_inference import ShadowModels
from art.utils import to_categorical
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased


from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


sys.path.insert(0, os.path.abspath('..'))
tf.compat.v1.disable_eager_execution()

#tf.random.set_seed(1234)
SEED=random.randint(0, 123494321)

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism()




#Initialization rule :
# if activation function is sigmoid or Tanh ==> use Glorot (Xavier or normalized Xavier)
# if activation is Relu ==> use He initialization from He et Al 2015

def Adult_NN(input_shape, init_distrib) :
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer = init_distrib, kernel_initializer= init_distrib)(inputs)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer= init_distrib, kernel_initializer= init_distrib )(x)
    x = tf.keras.layers.Dense(32, activation='relu', bias_initializer= init_distrib, kernel_initializer= init_distrib )(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='Recall'),
        tf.keras.metrics.Precision(name='Precision')

    ]

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
        #run_eagerly=True
    )
    return model


def train_from_model(model, x, y, epch, client_id) :
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    #10-Fold cross validation
    #kf = KFold(n_plits=10)
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=epch, verbose=1)
    plot_learningCurve(history, epch, client_id)
    model.evaluate(x_test, y_test)
    return model



def df_to_series(dataframe) :
    series = []
    for i in x_test.columns :
        series.append(x_train[i].squeeze())
    return pd.concat(series, axis=1)

def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

#model is a regressor ==> use 'loss' as input_types
#model is a Classifier ==> use 'prediction' as input_types
# works with attack_model_type = 'gb' (Gradient boosting) and attack_model_type = 'rf'  (random forest)
# error when attack_model_type ='nn' (Neural network) is used

def compute_PP2(x_train, model) :
    predictions = model.predict(x_train)
    return np.sum(np.where(predictions.flatten() >= 0.5, 1, 0))/x_train.shape[0]



def Shadow_models_MIA(model, x_train, y_train, x_test, y_test, x_shadow, y_shadow) :
    print('art classifier creation')
    art_classifier = KerasClassifier(model, use_logits=False)
    shadow_models = ShadowModels(art_classifier, num_shadow_models=3, disjoint_datasets=True, random_state=1)
    #Créer et entrainer les shadow sur x_shadow, y_shadow
    shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow.to_numpy(), y_shadow.to_numpy(), member_ratio=0.5)
    (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset
    # Shadow models' accuracy
    print('shadow models accuracy on test data : ', [sm.model.evaluate(x_test.to_numpy(), y_test.to_numpy()) for sm in shadow_models.get_shadow_models()])
    x_protected = x_test[x_test['sex']==1.0]
    y_protected = y_test[x_test['sex']==1.0]
    x_unprotected = x_test[x_test['sex']==0.0]
    y_unprotected = y_test[x_test['sex']==0.0]
    shadow_eods = []
    shadow_spds = []
    for sm in shadow_models.get_shadow_models() :
        eval_protected = sm.model.evaluate(x_protected.to_numpy(), y_protected.to_numpy())
        eval_unprotected = sm.model.evaluate(x_unprotected.to_numpy(), y_unprotected.to_numpy())
        shadow_eods.append(eval_protected[2] - eval_unprotected[2])
        shadow_spds.append(compute_PP2(x_protected, sm.model) - compute_PP2(x_unprotected, sm.model))
    # rf (random forest) or gb (gradient boosting)
    attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="gb")
    attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions, epochs=100)

    member_infer = attack.infer(x_train.to_numpy(), y_train.to_numpy())
    nonmember_infer = attack.infer(x_test.to_numpy(), y_test.to_numpy())
    member_acc = np.sum(member_infer) / len(x_train.to_numpy())
    nonmember_acc = 1 - np.sum(nonmember_infer) / len(x_test.to_numpy())
    acc = (member_acc * len(x_train.to_numpy()) + nonmember_acc * len(x_test.to_numpy())) / (len(x_train.to_numpy()) + len(x_test.to_numpy()))
    prec_recall = calc_precision_recall(np.concatenate((member_infer, nonmember_infer)),
                            np.concatenate((np.ones(len(member_infer)), np.zeros(len(nonmember_infer)))))

    return ([member_acc, nonmember_acc, acc], shadow_eods, shadow_spds, prec_recall)





if __name__ == '__main__':
    data = pd.read_csv('../datasets/adult.csv')

    attributes = {
            'race'      : data['race'].unique(),
            'sex'       : data['sex'].unique(),
         #   'relationship' : data['relationship'].unique(),
         #   'occupation' : data['occupation'].unique(),
            'marital.status' : data['marital.status'].unique(),
            'income' : data['income'].unique()
    }
    data.replace('?', np.NaN)
    print("Data preprocessing ...")
    pre_processed_data = data_PreProcess(data)

    #check if both have same distrib (necessary for Shokri et al attack).
    #check_balance(training_data, attributes)
    #check_balance(shadow_attack_data, attributes)

    y = pre_processed_data['income']
    x = pre_processed_data.drop('income', axis=1)


    input_shape = (x.shape[1],)

    print ("TRAIN/TEST splitting")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    #two options : Train the tensorflow model using tensorflow API fit() method then create the wrapper object KerasClassifier
    #              Create the KerasClassifier wrapper instance and call the wrapper fit() method
    #              Option 1 seems to provide a better attack performance
    #optimal accuracy for this dataset and model architecture
    epochs = 150
    #how to initilize kernel and bias weights
    init_distrib = tf.initializers.HeUniform(seed=SEED)
    scores = []
    model = Adult_NN(input_shape, init_distrib)
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=128, epochs=epochs, verbose=0)
    #plot_learningCurve(history, epochs, 1)

    #mesurer l'equité du model

    print('Target Model score over test data: ', model.evaluate(x_test, y_test))
    x_protected = x_test[x_test['sex']==1.0]
    y_protected = y_test[x_test['sex']==1.0]

    x_unprotected = x_test[x_test['sex']==0.0]
    y_unprotected = y_test[x_test['sex']==0.0]

    eval_protected = model.evaluate(x_protected, y_protected)
    eval_unprotected = model.evaluate(x_unprotected, y_unprotected)
    target_eod = eval_protected[2] - eval_unprotected[2]
    target_spd = compute_PP2(x_protected, model) - compute_PP2(x_unprotected, model)
    with open('attack_performance2.txt', 'w') as file:
        file.write(f"\nTarget Model score over protected group: {eval_protected}")
        file.write(f"\nTarget Model score over unprotected group:  {eval_unprotected}")
        file.write(f"\ntarget Model EOD :  {target_eod}")
        file.write(f"\ntarget Model SPD :  {target_spd}")


    eod_dists = []
    spd_dists = []
    prec = []
    recall = []
    member_acc = []
    nonmember_acc = []
    overall_acc = []

    #Apply the attack on the model.
    for epoch in [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2600] :
        for lmbd in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7] :
            dataset_path = 'new_tests5/checkpoint_samples_gan_eps='+str(lmbd)+'/epoch='+str(epoch)+'.csv'
            shadow_attack_data = pd.read_csv(dataset_path)
            y_shadow = shadow_attack_data['income']
            x_shadow = shadow_attack_data.drop('income', axis=1)

            ShadowAttack_score, shadow_eods, shadow_spds, prec_recall = Shadow_models_MIA(model, x_train, y_train, x_test, y_test, x_shadow, y_shadow)

            with open('attack_performance2.txt', 'a') as file:
                eod_dists.append(sum(shadow_eods)/3 - target_eod)
                spd_dists.append(sum(shadow_spds)/3 - target_spd)
                prec.append(prec_recall[0])
                recall.append(prec_recall[1])
                member_acc.append(ShadowAttack_score[0])
                nonmember_acc.append(ShadowAttack_score[1])
                overall_acc.append(ShadowAttack_score[2])
                file.write(f"\n\nAttacks summary for shadow-dataset : {dataset_path}")
                file.write(f"\nshadow models eods : {shadow_eods}. avg = {sum(shadow_eods)/3} dist to target_eod : {sum(shadow_eods)/3 - target_eod}")
                file.write(f"\nshadow models spds : {shadow_spds}. avg = {sum(shadow_spds)/3} dist to target_spd : {sum(shadow_spds)/3 - target_spd}")
                file.write('\n-------------Shadow Models (Shokri et Al) : --------------')
                file.write('\nMemebr_acc : '+ str(ShadowAttack_score[0]))
                file.write('\nNon-Memebr_acc : '+ str(ShadowAttack_score[1]))
                file.write('\nOverall acc : '+str(ShadowAttack_score[2]))
