# Comparative of HYBparsimony vs Bayesian Optimization and Feature Selection with Genetic Algorithms

Comparative_3Methods.ipynb notebook presents the results of comparing HYB-PARSIMONY with two other methods: Bayesian Optimization (BO) utilizing all features ($num\_cols$) and a classical three-step methodology based on GA for featuring selection:

- **Bayesian Optimization with all features (BO):** The column $bo\_FINAL\_SCORE\_TST$ represents the testing fitness value ($J_{tst}$) obtained using all input features ($bo\_NFS==num\_cols$),  $KernelRidge$ algorithm, and 250 iterations.

- **HYB-PARSIMONY:** The columns $hyb\_NFS$ and $hyb\_FINAL\_SCORE\_TST$ correspond to the number of features (NFS) and the fitness value ($J_{tst}$) obtained with each test dataset using HYB-PARSIMONY. HYB-PARSIMONY was executed with the following parameters: $\Gamma=0.50$, $nruns=250$, $time\_limit=120min$, $P=15$, $early\_{stopping}=35$, and $KernelRidge$ as the machine learning algorithm. Additionally, $ReRank$ was set to $0.001$, representing the maximum allowable difference between the $J$ values of two models to be considered equal.

- **SKLEARN-GENETIC-OPT with three steps:** The three-step methodology involves the following: first, performing hyperparameter optimization of $KernelRidge$ with BO using all features ($nruns=250$); second, employing Genetic Algorithms from the 'sklearn-genetic-opt' package for feature selection with the hyperparameters obtained in the first step and the following GA hyperparameters: $nruns=250$ and $P=15$; and finally, repeating the hyperparameter tuning with BO but using only the selected variables ($nruns=250$).The columns $ga\_NFS$ and $ga\_FINAL\_SCORE\_TST$ display the results obtained using $sklearn-genetic-opt$ with the three steps described above.

In these experiments, half of the instances from each dataset were used for training/validation, while the remaining half constituted the test dataset to assess the generalization capabilities of the models. The results represent the average values obtained from five runs of each methodology, each with different random seeds. 5-fold cross-validation was performed in all methods.

All the results are based on 100 datasets, covering binary ($44$), multiclass ($34$), and regression ($22$) problems, which were sourced from [openml.org](https://www.openml.org/) using the 'Download_Datasets.ipynb' notebook.


