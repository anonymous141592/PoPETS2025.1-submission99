# PoPETS2024-submission274

This GitHub repository contains the implementation of the #275 submission experiments to PoPETS2024.4 (4-th issue)

This repository contains two main directories : 

1) FHE-Fairness-awareFL : This directory contains the implementation of the specified homomorphic aggregation at section 6.

     As specified in the submission, this implementation is composed of two modules.
             -A Tensorflow v2 module that represents local training of the clients.
             -A Lattigo (FHE encryption library) module representing the server's homomorphic (With CKKS) fairness-aware aggregation.
     An intermediate directory "SharedFiles" serves to simulate the network, where clients write their updated models, and the server reads, encrypts then aggregates them.  

    -Dependencies :
       Numpy.
       Panda.
       Tensorflow.
       Lattigo homomorphic library.
       


3) Enhanced_MIA : This directory contains the implementation of the enhanced membership inference attack using prior knowledge of the target classifer's fairness level (a fairness metric evaluation). It implements the FairGAN architecture to generate parametrically biased data. This data later serves as input to train, and perform inferences for three shadow models. Finally, an attack classifier is trained to predict the membership status of a data-record.
 
    -Dependencies :
      Numpy.
      Panda.
      Tensorflow.
      Adverserial-Robustness-Toolbox.


*****************Datasets***********************
Three datasets are required to evaluate these experiments : 
1) Adult-Census-Income
2) Compas-Risk-Recidivism
3) FairFace


       
