This directory contains the implementation of enhanced membership attack presented at section 5.1.

FairGan.py and model_cond.py are the implementations of the parametrically biased data generation with the parameter lambda.
/Synthetic_data directory will contain all synthetic data generated via FairGan at several training checkpoints (every 200 epochs)
automated_test.sh lauches the training, of the Synthetizer, and data generation.

membership_attack.py implements the membership attack using Adverserial-robustness toolbox functions. Shadow models training's data is provided from the synthetic_data/ directory. Several datasets are tested for the shadow models training. Attack performances are then stored in log (text) files, along with the fairness of the shadow models, and their proximity to the fairness of the target classifier's fairness, showing better attack accuracies when the average fairness of the shadow models is close to the fairness of the target classifier. Hence, confirming the observation of Shokri et al. that the better the shadow models imitate the target model, the better the attack performance will be. We extend the imitation to the fairness.

Correlation_heatmaps.py implements Pearson's correaltion heatmap between data features of the synthetic datasets, in order to help visualize data unfairness.

MIA_plots.py contains vectors of sample results obtained from previous runs. And a plotting method.

-----------------------------------------------Attack adaptation--------------------------------------------------------
To adapt the attack to generic datasets, and situations. Tha main challenge is to carefully choose a value for the weight assigned to the second discriminator's loss (lambda). 


--------------------Choice of lambda-------------------
The FairGan architecture's goal is to debias the data. That is, create data-records with minimal disparate impact, and disparate treatment. For this purpose a second discriminator is integrated to the architecure. This second discriminator acts as fairness discriminator that predicts the sensitive attribute from a the non-sensitive ones. Therefore, the generator must produce samples where the sensitive attribute cannot be predicted from the non-sensitive ones. The loss of this second discriminator is scaled by a value lambda that gives "strength" (or not) to this second criteria. lambda = 0 is equivalent to a regular GAN (not a fair one). Higher values of indicate higher importance of the fairness constraint. 

To carefully choose lambda, one must understand that the amplitude of fairness his synthetic data is able to achieve is within a bounded range. Upper bounded by the pre-existing unfairness in the data that is fed to the GAN for training (Fairness can only improve, and not worsen with FairGAN, as negative values of lambda yield unstable behaviour of the GAN). And lower-bounded by a minimal value close to 0 for very high values of lambda (e.g., 5, 6, 7 for Adult). However, very high values of lambda degrade data's utility (making a classifier trained on the synthectic data poorly generalize to real data). Nevertheless, for training shadow models, accuracy is not necessary, as these models will only serve to mimic the target classifer's behaviour, and create an attack database, and not to perform predictions. So given the level of unfairness already present in the attacker's data (and also its nature, as FairGan is more efficient in removing disparate impact), and the target classifier's fairness level. The attacker chooses lambda that provides closest fairness value to the intended one.












