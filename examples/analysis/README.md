# Comparative of HYBparsimony vs Bayesian Optimization and method with Feature Selection using Genetic Algorithms

[Comparative_3Methods.ipynb](Comparative_3Methods.ipynb) notebook presents the results of comparing HYB-PARSIMONY with two other methods: Bayesian Optimization (BO) utilizing all features (*num\_cols*) and a classical three-step methodology based on GA for featuring selection:

- Bayesian Optimization with all features (BO).
- HYB-PARSIMONY
- SKLEARN-GENETIC-OPT with three steps: The three-step methodology involves the following: first, performing hyperparameter optimizationwith BO using all features ($nruns=250$); second, employing Genetic Algorithms from the 'sklearn-genetic-opt' package for feature selection with the hyperparameters obtained in the first step; and finally, repeating the hyperparameter tuning with BO but using only the selected variables.

In these experiments, half of the instances from each dataset were used for training/validation, while the remaining half constituted the test dataset to assess the generalization capabilities of the models. The results represent the average values obtained from five runs of each methodology, each with different random seeds. 5-fold cross-validation was performed in all methods.

All the results are based on 100 datasets, covering binary ($44$), multiclass ($34$), and regression ($22$) problems, which were sourced from [openml.org](https://www.openml.org/) using the 'Download_Datasets.ipynb' notebook.

To replicate results:

1. Download datasets with 'Download_Datasets.ipynb' notebook.
2. Execute 'HybComparative.py' to obtain results with HYB_PARSIMONY.
3. Execute 'BOComparative.py' to obtain results with Bayesian Optimization.
4. Execute 'SKGENETICSComparative.py' to obtain results with HYB_PARSIMONY.
5. See results with 'Comparative_3Methods.ipynb' notebook.

   


