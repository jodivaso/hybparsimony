# -*- coding: utf-8 -*-

"""hybparsimony for Python is a package for searching accurate parsimonious models by combining feature selection (FS), model
hyperparameter optimization (HO), and parsimonious model selection (PMS) based on a separate cost and complexity evaluation.

To improve the search for parsimony, the hybrid method combines GA mechanisms such as selection, crossover and mutation within a PSO-based optimization algorithm that includes a strategy in which the best position of each particle (thus also the best position of each neighborhood) is calculated taking into account not only the goodness-of-fit, but also the parsimony principle. 

In hybparsimony, the percentage of variables to be replaced with GA at each iteration $t$ is selected by a decreasing exponential function:
 $pcrossover=max(0.80 \cdot e^{(-\Gamma \cdot t)}, 0.10)$, that is adjusted by a $\Gamma$ parameter (by default $\Gamma$ is set to $0.50$). Thus, in the first iterations parsimony is promoted by GA mechanisms, i.e., replacing by crossover a high percentage of particles at the beginning. Subsequently, optimization with PSO becomes more relevant for the improvement of model accuracy. This differs from other hybrid methods in which the crossover is applied between the best individual position of each particle or other approaches in which the worst particles are also replaced by new particles, but at extreme positions.

Experiments show that, in general, and with a suitable $\Gamma$, hybparsimony allows to obtain better, more parsimonious and more robust models compared to other methods. It also reduces the number of iterations and, consequently, the computational effort.

References
----------
Divasón, J., Pernia-Espinoza, A., Martinez-de-Pison, F.J. (2022).
New Hybrid Methodology Based on Particle Swarm Optimization with Genetic Algorithms to Improve 
the Search of Parsimonious Models in High-Dimensional Databases.
In: García Bringas, P., et al. 
Hybrid Artificial Intelligent Systems. HAIS 2022. 
Lecture Notes in Computer Science, vol 13469. Springer, Cham.
[https://doi.org/10.1007/978-3-031-15471-3_29](https://doi.org/10.1007/978-3-031-15471-3_29)
"""

from GAparsimony import Population, order
from GAparsimony.util import parsimony_monitor, parsimony_summary
from GAparsimony.lhs import geneticLHS, improvedLHS, maximinLHS, optimumLHS, randomLHS

import warnings
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import time
import inspect, opcode




class GAparsimony(object):

    def __init__(self, 
                fitness,
                params,
                features,
                type_ini_pop="improvedLHS", 
                popSize = 50, 
                pcrossover = 0.8,  
                maxiter = 40, 
                feat_thres=0.90, 
                rerank_error = 0.0, 
                iter_start_rerank = 0,
                pmutation = 0.10, 
                feat_mut_thres=0.10, 
                not_muted=3,
                tol = 1e-4,
                elitism = None,
                selection = "nlinear", 
                keep_history = False,
                early_stop = None, 
                maxFitness = np.Inf, 
                suggestions = None,
                seed_ini = None, 
                verbose=1):
        r"""
        A class for searching parsimonious models by feature selection and parameter tuning with
        genetic algorithms.

        Parameters
        ----------
        fitness : function
            The fitness function, any function which takes as input a chromosome which combines the model parameters 
            to tune and the features to be selected. Fitness function returns a numerical vector with three values:validation_cost, 
            testing_cost and model_complexity, and the trained model. 
        params : dict
            It is a dictionary with the model's hyperparameters to be adjusted and the range of values to search for.

            .. code-block::

                {
                    "<< hyperparameter name >>": 
                    {
                        "range": [<< minimum value >>, << maximum value >>],
                        "type": GAparsimony.FLOAT/GAparsimony.INTEGER
                    },
                    "<< hyperparameter name >>": 
                    {
                        "value": << constant value >>,
                        "type": GAparsimony.CONSTANT
                    }
                }

        features : int or list of str
            The number of features/columns in the dataset or a list with their names.
        type_ini_pop : str, {'randomLHS', 'geneticLHS', 'improvedLHS', 'maximinLHS', 'optimumLHS', 'random'}, optional
            Method to create the first population with `GAparsimony._population` function. Possible values: `randomLHS`, `geneticLHS`, 
            `improvedLHS`, `maximinLHS`, `optimumLHS`, `random`.First 5 methods correspond with several latine hypercube for initial sampling. By default is set to `improvedLHS`.
        popSize : int, optional
            The population size.
        pcrossover : float, optional
            The probability of crossover between pairs of chromosomes. Typically this is alarge value and by default is set to `0.8`.
        maxiter : float, optional
            The maximum number of iterations to run before the GA process is halted.
        feat_thres : float, optional
            Proportion of selected features in the initial population. It is recommended a high percentage of the selected features for 
            the first generations. By default is set to `0.90`.
        rerank_error : float, optional
            When a value is provided, a second reranking process according to the model complexities is called by `parsimony_rerank` function. 
            Its primary objective isto select individuals with high validation cost while maintaining the robustnessof a parsimonious model. 
            This function switches the position of two models if the first one is more complex than the latter and no significant difference 
            is found between their fitness values in terms of cost. Thus, if the absolute difference between the validation costs are 
            lower than `rerank_error` they areconsidered similar. Default value=`0.01`.
        iter_start_rerank : int, optional
            Iteration when ReRanking process is actived. Default=`0`. Sometimes is useful not to use ReRanking process in the first generations.
        pmutation : float, optional
            The probability of mutation in a parent chromosome. Usually mutation occurswith a small probability. By default is set to `0.10`.
        feat_mut_thres : float, optional
            Probability of the muted `features-chromosome` to be one. Default value is set to `0.10`.
        not_muted : int, optional
            Number of the best elitists that are not muted in each generation. Default valueis set to `3`.
        elitism : int, optional
            The number of best individuals to survive at each generation. By default the top `20%` individuals will survive at each iteration.
        selection : str, optional
            Method to perform selection with `GAparsimony._selection` function. Possible values: `linear`, `nlinear`, `random`. By default is set to `nlinear`.
        keep_history : bool, optional
            If it is `True` keeps in the list GAparsimony.history each generation as `pandas.DataFrame`. This parameter must set `True` in 
            order to use `GAparsimony.plot` method or `GAparsimony.importance` function.
        maxFitness : int, optional
            The upper bound on the fitness function after that the GA search is interrupted. Default value is set to +Inf.
        early_stop : int, optional
            The number of consecutive generations without any improvement in the bestfitness value before the GA is stopped.
        suggestions : numpy.array, optional
            A matrix of solutions strings to be included in the initial population.
        seed_ini : int, optional
            An integer value containing the random number generator state.
        verbose : int, optional
            The level of messages that we want it to show us. Possible values: 1=monitor level,  2=debug level, if 0 no messages. Default `1`.


        Attributes
        ----------
        population : Population
            The current (or final) population.
        minutes_total : float
            Total elapsed time (in minutes).
        history : float
            A list with the population of all iterations.
        best_score : float
            The best validation score in the whole GA process.
        best_model
            The best model in the whole GA process.
        best_model_conf : Chromosome
            The parameters and features of the best model in the whole GA process.
        bestfitnessVal : float
            The validation cost of the best solution at the last iteration.
        bestfitnessTst : float
            The testing cost of the best solution at the last iteration.
        bestcomplexity : float
            The model complexity of the best solution at the last iteration.

        Examples
        --------
        Usage example for a regression model using the sklearn boston dataset 

        .. highlight:: python
        .. code-block:: python

            from sklearn.model_selection import RepeatedKFold
            from sklearn.linear_model import Lasso
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_squared_error

            from sklearn.datasets import load_boston

            from GAparsimony import GAparsimony, Population, getFitness
            from GAparsimony.util import linearModels_complexity

            boston = load_boston()
            X, y = boston.data, boston.target 
            X = StandardScaler().fit_transform(X)

            # ga_parsimony can be executed with a different set of 'rerank_error' values
            rerank_error = 0.01

            params = {"alpha":{"range": (1., 25.9), "type": Population.FLOAT}, 
                        "tol":{"range": (0.0001,0.9999), "type": Population.FLOAT}}

            cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

            fitness = getFitness(Lasso, mean_squared_error, linearModels_complexity, cv, minimize=True, 
                                 test_size=0.2, random_state=42, n_jobs=-1)


            GAparsimony_model = GAparsimony(fitness=fitness,
                                        params = params, 
                                        features = boston.feature_names,
                                        keep_history = True,
                                        rerank_error = rerank_error,
                                        popSize = 40,
                                        maxiter = 50, early_stop=10,
                                        feat_thres=0.90, # Perc selected features in first generation
                                        feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                        seed_ini = 1234)


            GAparsimony_model.fit(X, y)

            GAparsimony_model.summary()

            aux = GAparsimony_model.summary()

            GAparsimony_model.plot()

        .. code-block:: text

            GA-PARSIMONY | iter = 0
            MeanVal = -79.1813338 | ValBest = -30.3470614 | TstBest = -29.2466835 |ComplexBest = 13000000021.927263|  Time(min) = 0.185549

            GA-PARSIMONY | iter = 1
            MeanVal = -55.0713465 | ValBest = -30.2283235 | TstBest = -29.2267507 |ComplexBest = 12000000022.088743| Time(min) = 0.1238126  

            GA-PARSIMONY | iter = 2
            MeanVal = -34.8473723 | ValBest = -30.2283235 | TstBest = -29.2267507 |ComplexBest = 12000000022.088743| Time(min) = 0.0907046  

            GA-PARSIMONY | iter = 3
            MeanVal = -38.5251529 | ValBest = -30.0455259 | TstBest = -29.2712578 |ComplexBest = 10000000022.752678| Time(min) = 0.0755356  

            ...

            GA-PARSIMONY | iter = 20
            MeanVal = -34.2636095 | ValBest = -29.5036901 | TstBest = -29.3245069 |ComplexBest = 5000000023.115818| Time(min) = 0.0659549  

            GA-PARSIMONY | iter = 21
            MeanVal = -40.4629864 | ValBest = -29.5036901 | TstBest = -29.3245069 |ComplexBest = 5000000023.115818| Time(min) = 0.0725066  

            GA-PARSIMONY | iter = 22
            MeanVal = -35.9230384 | ValBest = -29.5036901 | TstBest = -29.3245069 |ComplexBest = 5000000023.115818| Time(min) = 0.0704362  

            GA-PARSIMONY | iter = 23
            MeanVal = -36.5946762 | ValBest = -29.5036901 | TstBest = -29.3245069 |ComplexBest = 5000000023.115818| Time(min) = 0.0723252  

            GA-PARSIMONY | iter = 24
            MeanVal = -37.3293511 | ValBest = -29.5036901 | TstBest = -29.3245069 |ComplexBest = 5000000023.115818| Time(min) = 0.0684883  

            +------------------------------------+
            |             GA-PARSIMONY           |
            +------------------------------------+

            GA-PARSIMONY settings:
            Number of Parameters      = 2
            Number of Features        = 13
            Population size           = 40
            Maximum of generations    = 50
            Number of early-stop gen. = 10
            Elitism                   = 8
            Crossover probability     = 0.8
            Mutation probability      = 0.1
            Max diff(error) to ReRank = 0.01
            Perc. of 1s in first popu.= 0.9
            Prob. to be 1 in mutation = 0.1

            Search domain =
                    alpha     tol  CRIM   ZN  INDUS  CHAS  NOX   RM  AGE  DIS  RAD  \
            Min_param    1.0  0.0001   0.0  0.0    0.0   0.0  0.0  0.0  0.0  0.0  0.0
            Max_param   25.9  0.9999   1.0  1.0    1.0   1.0  1.0  1.0  1.0  1.0  1.0

                    TAX  PTRATIO    B  LSTAT
            Min_param  0.0      0.0  0.0    0.0
            Max_param  1.0      1.0  1.0    1.0


            GA-PARSIMONY results:
            Iterations                = 25
            Best validation score = -29.502012171608403


            Solution with the best validation score in the whole GA process =

            fitnessVal fitnessTst complexity    alpha       tol CRIM ZN INDUS CHAS NOX  \
            0    -29.502   -29.3244      5e+09  1.33694  0.541197    0  0     0    0   0

            RM AGE DIS RAD TAX PTRATIO  B LSTAT
            0  1   1   0   0   0       1  1     1


            Results of the best individual at the last generation =

            Best indiv's validat.cost = -29.503690126221098
            Best indiv's testing cost = -29.324506895493244
            Best indiv's complexity   = 5000000023.115818
            Elapsed time in minutes   = 2.031593410174052


            BEST SOLUTION =

            fitnessVal fitnessTst complexity   alpha       tol CRIM ZN INDUS CHAS NOX  \
            0   -29.5037   -29.3245      5e+09  1.3374  0.547189    0  0     0    0   0

            RM AGE DIS RAD TAX PTRATIO  B LSTAT
            0  1   1   0   0   0       1  1     1
        
        .. figure:: ../docs/img/regression.png
            :align: center
            :width: 600
            :alt: Regression plot

            Regression plot
        
        Usage example for a classification model using the wine dataset 

        .. highlight:: python
        .. code-block:: python

            from sklearn.model_selection import RepeatedKFold
            from sklearn.svm import SVC
            from sklearn.metrics import cohen_kappa_score
            from sklearn.datasets import load_wine

            from GAparsimony import GAparsimony, Population, getFitness
            from GAparsimony.util import svm_complexity

            wine = load_wine()
            X, y = wine.data, wine.target 
            X = StandardScaler().fit_transform(X)

            rerank_error = 0.001
            params = {"C":{"range": (00.0001, 99.9999), "type": Population.FLOAT}, 
                    "gamma":{"range": (0.00001,0.99999), "type": Population.FLOAT}, 
                    "kernel": {"value": "poly", "type": Population.CONSTANT}}

            cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

            fitness = getFitness(SVC, cohen_kappa_score, svm_complexity, cv, minimize=False, test_size=0.2, random_state=42, n_jobs=-1)

            GAparsimony_model = GAparsimony(fitness=fitness,
                                            params=params,
                                            features=wine.feature_names,
                                            keep_history = True,
                                            rerank_error = rerank_error,
                                            popSize = 40,
                                            maxiter = 50, early_stop=10,
                                            feat_thres=0.90, # Perc selected features in first generation
                                            feat_mut_thres=0.10, # Prob of a feature to be one in mutation
                                            seed_ini = 1234)



            GAparsimony_model.fit(X, y)

            GAparsimony_model.summary()

            GAparsimony_model.plot()

        .. code-block:: text

            GA-PARSIMONY | iter = 0
            MeanVal = 0.879549  |  ValBest = 0.9314718  |  TstBest = 0.9574468  |ComplexBest = 10000000045.0| Time(min) = 0.1438692  

            GA-PARSIMONY | iter = 1
            MeanVal = 0.9075035  |  ValBest = 0.9496819  |  TstBest = 0.9142857  |ComplexBest = 11000000060.0| Time(min) = 0.0893566  

            GA-PARSIMONY | iter = 2
            MeanVal = 0.9183232  |  ValBest = 0.9496819  |  TstBest = 0.9142857  |ComplexBest = 11000000060.0| Time(min) = 0.0818844  

            GA-PARSIMONY | iter = 3
            MeanVal = 0.9219764  |  ValBest = 0.9534295  |  TstBest = 0.9568345  |ComplexBest = 10000000043.0| Time(min) = 0.0739248 

            ...

            GA-PARSIMONY | iter = 19
            MeanVal = 0.9182586  |   ValBest = 0.972731  |     TstBest = 1.0     |ComplexBest = 7000000048.0| Time(min) = 0.0867344  

            GA-PARSIMONY | iter = 20
            MeanVal = 0.9224294  |   ValBest = 0.972731  |     TstBest = 1.0     |ComplexBest = 7000000048.0| Time(min) = 0.0771279  

            GA-PARSIMONY | iter = 21
            MeanVal = 0.9150223  |   ValBest = 0.972731  |     TstBest = 1.0     |ComplexBest = 7000000048.0| Time(min) = 0.0847196  

            GA-PARSIMONY | iter = 22
            MeanVal = 0.9335024  |   ValBest = 0.972731  |     TstBest = 1.0     |ComplexBest = 7000000048.0| Time(min) = 0.0814945 


            +------------------------------------+
            |             GA-PARSIMONY           |
            +------------------------------------+

            GA-PARSIMONY settings:
            Number of Parameters      = 2
            Number of Features        = 13
            Population size           = 40
            Maximum of generations    = 50
            Number of early-stop gen. = 10
            Elitism                   = 8
            Crossover probability     = 0.8
            Mutation probability      = 0.1
            Max diff(error) to ReRank = 0.001
            Perc. of 1s in first popu.= 0.9
            Prob. to be 1 in mutation = 0.1

            Search domain =
                            C    gamma  alcohol  malic_acid  ash  alcalinity_of_ash  \
            Min_param   0.0001  0.00001      0.0         0.0  0.0                0.0
            Max_param  99.9999  0.99999      1.0         1.0  1.0                1.0

                    magnesium  total_phenols  flavanoids  nonflavanoid_phenols  \
            Min_param        0.0            0.0         0.0                   0.0
            Max_param        1.0            1.0         1.0                   1.0

                    proanthocyanins  color_intensity  hue  \
            Min_param              0.0              0.0  0.0
            Max_param              1.0              1.0  1.0

                    od280/od315_of_diluted_wines  proline
            Min_param                           0.0      0.0
            Max_param                           1.0      1.0


            GA-PARSIMONY results:
            Iterations                = 23
            Best validation score = 0.9727309855126027


            Solution with the best validation score in the whole GA process =

            fitnessVal fitnessTst complexity        C      gamma alcohol malic_acid ash  \
            0   0.972731          1      7e+09  51.1573  0.0581044       1          0   1

            alcalinity_of_ash magnesium total_phenols flavanoids nonflavanoid_phenols  \
            0                 1         0             0          1                    0

            proanthocyanins color_intensity hue od280/od315_of_diluted_wines proline
            0               0               0   1                            1       1


            Results of the best individual at the last generation =

            Best indiv's validat.cost = 0.9727309855126027
            Best indiv's testing cost = 1.0
            Best indiv's complexity   = 7000000048.0
            Elapsed time in minutes   = 1.9634766817092892


            BEST SOLUTION =

            fitnessVal fitnessTst complexity        C      gamma alcohol malic_acid ash  \
            0   0.972731          1      7e+09  51.1573  0.0581044       1          0   1

            alcalinity_of_ash magnesium total_phenols flavanoids nonflavanoid_phenols  \
            0                 1         0             0          1                    0

            proanthocyanins color_intensity hue od280/od315_of_diluted_wines proline
            0               0               0   1                            1       1
        
        .. figure:: ../docs/img/classification.png
            :align: center
            :width: 600
            :alt: Classification plot

            Classification plot
        """
        
        self.elitism = max(1, round(popSize * 0.20)) if not elitism else elitism

        self.population = Population(params, columns=features)


        # Check parameters
        # ----------------
        if selection is None or type(selection) is not str:
            raise ValueError("A selection(sting) must be provided!!!")
        if fitness is None:
            raise Exception("A fitness function must be provided!!!")
        if not callable(fitness):
            raise Exception("A fitness function must be provided!!!")
        if popSize < 10:
            warnings.warn("The population size is less than 10!!!")
        if maxiter < 1:
            raise ValueError("The maximum number of iterations must be at least 1!!!")
        if self.elitism > popSize:
            raise ValueError("The elitism cannot be larger that population size.")
        if pcrossover < 0 or pcrossover > 1:
            raise ValueError("Probability of crossover must be between 0 and 1!!!")
        if pmutation < 0 or pmutation > 1:
            raise ValueError("Probability of mutation must be between 0 and 1!!!")
        if self.population._min is None and self.population._max is None:
            raise ValueError("A min and max range of values must be provided!!!")
        if self.population._min.shape != self.population._max.shape:
            raise Exception("min_param and max_param must have the same length!!!")
        if features is None:
            raise Exception("Number of features or name of features must be provided!!!")
        if (suggestions is not None) or (type(suggestions) is list and len(suggestions)>0 and type(suggestions[0]) is not list) or (type(suggestions) is np.array and len(suggestions.shape) < 2):
            raise Exception("Provided suggestions is a vector")
        if (type(suggestions) is np.array) and (self.population._min.shape + features) != suggestions.shape[1]:
            raise Exception("Provided suggestions (ncol) matrix do not match the number of variables (model parameters + vector with selected features) in the problem!")

        self.fitness = fitness
        self.popSize = popSize
        self.pcrossover = pcrossover
        self.maxiter = maxiter
        self.feat_thres=feat_thres
        self.rerank_error=rerank_error
        self.iter_start_rerank=iter_start_rerank
        self.pmutation = pmutation
        self.feat_mut_thres=feat_mut_thres
        self.not_muted=not_muted
        self.tol = tol
        
        self.selection = selection
        self.keep_history = keep_history
        self.early_stop = maxiter if not early_stop else early_stop
        self.maxFitness = maxFitness
        self.suggestions = np.array(suggestions) if type(suggestions) is list else suggestions
        self.seed_ini = seed_ini
        self.verbose = verbose

        self.iter = 0
        self.minutes_total=0
        self.best_score = np.NINF
        self.best_complexity = np.Inf
        self.best_parsimony_score = np.NINF
        self.best_parsimony_complexity = np.Inf
        self.history = list()

        if self.seed_ini:
            np.random.seed(self.seed_ini)

        self.population.population = self._population(type_ini_pop=type_ini_pop) # Initial population

        # Update initial population to satisfy the initial feat_thres
        self.population.update_to_feat_thres(self.popSize, self.feat_thres)


        if self.suggestions:
            ng = min(self.suggestions.shape[0], popSize)
            if ng > 0:
                self.population[:ng, :] = self.suggestions[:ng, :]
    
    def fit(self, X, y, iter_ini=0):
        r"""
        A GA-based optimization method for searching accurate parsimonious models by combining feature selection, model tuning, 
        and parsimonious model selection (PMS). PMS procedure is basedon separate cost and complexity evaluations. The best individuals 
        are initially sorted by an *errorfitness* function, and afterwards, models with similar costs are rearranged according to their 
        modelcomplexity so as to foster models of lesser complexity.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.array
            Training vector.
        y : pandas.DataFrame or numpy.array
            Target vector relative to X.
        iter_ini : int, optional
            Iteration/generation of `GAparsimony.history` to be used when model is pretrained. If
            `iter_ini==None` uses the last iteration of the model.

        """
           
        # Get suggestions
        # ---------------
        if self.verbose == 2 and self.suggestions:
            print(self.suggestions)


        # Initial settings
        # ----------------
        if self.seed_ini:
            np.random.seed(self.seed_ini)

        self._summary = np.empty((self.maxiter,6*3,))
        self._summary[:] = np.nan
        self.bestSolList = list()
        self.fitnessval = np.empty(self.popSize)
        self.fitnessval[:] = np.nan
        self.fitnesstst = np.empty(self.popSize)
        self.fitnesstst[:] = np.nan
        self.complexity = np.empty(self.popSize)
        self.complexity[:] = np.nan
        _models = np.empty(self.popSize).astype(object)


        if len(self.history) > 0:
            if self.verbose == 2:
                print("There is a GAparsimony 'object'!!!")
                print(self)

            iter_ini = self.iter if not iter_ini else min(iter_ini, self.iter)
            if iter_ini < 0:
                iter_ini = 0

            self.history = self.history[iter_ini].values[0]
            if self.verbose == 2:
                print(f"Starting GA optimization with a provided GAparsimony 'object'. Using object's GA settings and its population from iter={iter_ini}.")
        
        elif self.verbose == 2:
            print("\nStep 0. Initial population")
            print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
            # input("Press [enter] to continue")


        # Main Loop
        # --------- 
        for iter in range(self.maxiter):
            tic = time.time()
            
            self.iter = iter

            for t in range(self.popSize):
                c = self.population.getChromosome(t)
                if np.isnan(self.fitnessval[t]) and np.sum(c.columns)>0:
                    fit = self.fitness(c, X=X, y=y)
                    self.fitnessval[t] = fit[0][0]
                    self.fitnesstst[t] = fit[0][1]
                    self.complexity[t] = fit[0][2]
                    _models[t] = fit[1]


            if self.seed_ini:
                np.random.seed(self.seed_ini*iter) 
            

            # Sort by the Fitness Value
            # ----------------------------
            sort = order(self.fitnessval, kind='heapsort', decreasing = True, na_last = True)
            self.population.population = self.population[sort, :]
            self.fitnessval = self.fitnessval[sort]
            self.fitnesstst = self.fitnesstst[sort]
            self.complexity = self.complexity[sort]
            _models = _models[sort]

            PopSorted = self.population._pop.copy()
            FitnessValSorted = self.fitnessval.copy()
            FitnessTstSorted = self.fitnesstst.copy()
            ComplexitySorted = self.complexity.copy()
            
            if np.max(self.fitnessval)>self.best_score:
                self.best_score = np.nanmax(self.fitnessval)
                self.solution_best_score = np.r_[self.best_score, 
                                                self.fitnesstst[np.argmax(self.fitnessval)], 
                                                self.complexity[np.argmax(self.fitnessval)], 
                                                self.population.population[np.argmax(self.fitnessval)]]


            if self.verbose == 2:
                print("\nStep 1. Fitness sorted")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                # input("Press [enter] to continue")

            
            # Reorder models with ReRank function
            # -----------------------------------
            if self.rerank_error != 0.0 and self.iter >= self.iter_start_rerank:
                ord_rerank = self._rerank()
                self.population.population = self.population[ord_rerank]
                self.fitnessval = self.fitnessval[ord_rerank]
                self.fitnesstst = self.fitnesstst[ord_rerank]
                self.complexity = self.complexity[ord_rerank]
                _models = _models[ord_rerank]

                PopSorted = self.population._pop.copy()
                FitnessValSorted = self.fitnessval.copy()
                FitnessTstSorted = self.fitnesstst.copy()
                ComplexitySorted = self.complexity.copy()
                
                if self.verbose == 2:
                    print("\nStep 2. Fitness reranked")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                    # input("Press [enter] to continue")


            # Keep results
            # ---------------
            self._summary[iter, :] = parsimony_summary(self) # CAMBIAR AL CAMBIAR FUNCION PARSIMONY_SUMARY
            

            # Keep Best Solution
            # ------------------
            self.bestfitnessVal = self.fitnessval[0]
            self.bestfitnessTst = self.fitnesstst[0]
            self.bestcomplexity = self.complexity[0]
            self.bestsolution = np.concatenate([[self.bestfitnessVal, self.bestfitnessTst, self.bestcomplexity],self.population.population[0]])
            self.bestSolList.append(self.bestsolution)

            # Keep Best Parsimony Solution
            bestParsimonyFitnessVal = FitnessValSorted[1]
            bestParsimonyFitnessTst = FitnessTstSorted[1]
            bestParsimonyComplexity = ComplexitySorted[1]
            bestIterParsimonySolution = np.concatenate(
                [[bestParsimonyFitnessVal, bestParsimonyFitnessTst, bestParsimonyComplexity], PopSorted[1]])

            # Keep Best Model
            # ------------------
            self.best_model = _models[0]
            self.best_model_conf = self.population.getChromosome(0)

            # Keep Global best parsimony model

            # Si el mejor de la iteración mejora el parsimonioso global y es más simple, actualizo.
            # Si el mejor de la iteración mejora el parsimonioso global y es más complejo, NO actualizo (ya estará guardado en self.best_score)
            # Si el mejor parsimonios de la iteración mejora el parsimonioso global y es más simple, actualizo
            # Si el mejor parsimonioso de la iteración mejora el parsimonioso global y es más complejo,
            # entonces aplico algo parecido al rerank (lo actualizo si mejora mucho, aunque sea más complejo)

            if self.bestfitnessVal >= self.best_parsimony_score and self.bestcomplexity < self.best_parsimony_complexity:
                self.best_parsimony_score = self.bestfitnessVal
                self.best_parsimony_solution = self.bestsolution
                self.best_parsimony_complexity = self.bestcomplexity
                self.solution_best_parsimony_score = np.r_[self.best_score,
                                                           self.bestfitnessVal,
                                                           self.bestfitnessTst,
                                                           self.bestcomplexity]
                self.best_parsimony_model = _models[0]
                self.best_parsimony_model_conf = PopSorted[0]

            if (bestParsimonyFitnessVal >= self.best_parsimony_score and bestParsimonyComplexity < self.best_parsimony_complexity) \
                    or (bestParsimonyFitnessVal > self.best_parsimony_score + self.rerank_error):
                self.best_parsimony_score = bestParsimonyFitnessVal
                self.best_parsimony_solution = bestIterParsimonySolution
                self.best_parsimony_complexity = bestParsimonyComplexity
                self.solution_best_parsimony_score = np.r_[self.best_parsimony_score,
                                                           bestParsimonyFitnessVal,
                                                           bestParsimonyFitnessTst,
                                                           bestParsimonyComplexity]
                self.best_parsimony_model = _models[1]
                self.best_parsimony_model_conf = PopSorted[1]

            # Keep elapsed time in minutes
            # ----------------------------
            tac = time.time()
            self.minutes_gen = (tac - tic) / 60.0
            self.minutes_total = self.minutes_total+self.minutes_gen
            

            # Keep this generation into the History list
            # ------------------------------------------
            if self.keep_history:
                self.history.append(pd.DataFrame(np.c_[self.population.population, self.fitnessval, self.fitnesstst, self.complexity], columns=list(self.population._params.keys())+self.population.colsnames+["fitnessval", "fitnesstst", "complexity"]))
            

            # Call to 'monitor' function
            # --------------------------
            if self.verbose > 0:
                parsimony_monitor(self)  
            
            if self.verbose == 2:
                print("\nStep 3. Fitness results")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                # input("Press [enter] to continue")
            
            
            # Exit?
            # -----
            best_val_cost = self._summary[:,0][~np.isnan(self._summary[:,0])]
            if self.bestfitnessVal >= self.maxFitness:
                break
            if self.iter == self.maxiter:
                break
            if (len(best_val_cost) - (np.min(np.arange(len(best_val_cost))[best_val_cost >= (
                    np.max(best_val_cost) - self.tol)]))) >= self.early_stop:
                break

            
            # Selection Function
            # ------------------
            self._selection()

            if self.verbose == 2:
                print("\nStep 4. Selection")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                # input("Press [enter] to continue")


            # CrossOver Function
            # ------------------
            if self.pcrossover > 0:
                nmating = int(np.floor(self.popSize/2))
                mating = np.random.choice(list(range(2 * nmating)), size=(2 * nmating), replace=False).reshape((nmating, 2))
                for i in range(nmating):
                    if self.pcrossover > np.random.uniform(low=0, high=1):
                        parents = mating[i, ]
                        self._crossover(parents=parents)
                if self.verbose == 2:
                    print("\nStep 5. CrossOver")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                    # input("Press [enter] to continue")


            # New generation with elitists
            # ----------------------------
            if (self.elitism > 0):
                self.population[:self.elitism] = PopSorted[:self.elitism]
                self.fitnessval[:self.elitism] = FitnessValSorted[:self.elitism]
                self.fitnesstst[:self.elitism] = FitnessTstSorted[:self.elitism]
                self.complexity[:self.elitism] = ComplexitySorted[:self.elitism]
            if (self.verbose == 2):
                print("\nStep 6. With Elitists")
                print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                # input("Press [enter] to continue")
            

            # Mutation function
            # -----------------
            if self.pmutation > 0:
                self._mutation()
                if self.verbose == 2:

                    print("\nStep 7. Mutation")
                    print(np.c_[self.fitnessval, self.fitnesstst, self.complexity, self.population.population][:10, :])
                    # input("Press [enter] to continue")
    
    def _rerank(self):
        r"""
        Function for reranking by complexity in parsimonious model selectionprocess. Promotes models with similar fitness but lower complexity to top positions.

        This method corresponds with the second step of parsimonious model selection (PMS) procedure.PMS works in the 
        following way: in each GA generation, best solutions are first sorted by their cost,J. Then, in a second step, 
        individuals with less complexity are moved to the top positions when theabsolute difference of their J is lower 
        than aobject@rerank_errorthreshold value. Therefore, theselection of less complex solutions among those with similar 
        accuracy promotes the evolution ofrobust solutions with better generalization capabilities.

        Returns
        -------
        numpy.array
            A vector with the new position of the individuals

        """

        cost1 = self.fitnessval.copy().astype(float)
        cost1[np.isnan(cost1)]= np.NINF

        sort = order(cost1, decreasing = True)
        cost1 = cost1[sort]
        complexity = self.complexity.copy()
        complexity[np.isnan(complexity)] = np.Inf
        complexity = complexity[sort]
        position = sort
  
        # start
        pos1 = 1
        pos2 = 2
        cambio = False
        error_posic = cost1[pos1]
  
        while not pos1 == self.popSize:
            # Obtaining errors
            if pos2 >= self.popSize:
                if cambio:
                    pos2 = pos1+1
                    cambio = False
                else:
                    break
            error_indiv2 = cost1[pos2]
    
            # Compare error of first individual with error_posic. Is greater than threshold go to next point
            #      if ((Error.Indiv1-error_posic) > model@rerank_error) error_posic=Error.Indiv1

            if np.isfinite(error_indiv2) and np.isfinite(error_posic):
                error_dif = abs(error_indiv2 - error_posic)
            else:
                error_dif = np.Inf

            if error_dif < self.rerank_error:
        
                # If there is not difference between errors swap if Size2nd < SizeFirst
                size_indiv1 = complexity[pos1]
                size_indiv2 = complexity[pos2]
                if size_indiv2<size_indiv1:
            
                    cambio = True
                
                    swap_indiv = cost1[pos1]
                    cost1[pos1] = cost1[pos2]
                    cost1[pos2] = swap_indiv
                            
                    complexity[pos1], complexity[pos2] = complexity[pos2], complexity[pos1]

                    position[pos1], position[pos2] = position[pos2], position[pos1]
                
                    if self.verbose == 2:
                        print(f"SWAP!!: pos1={pos1}({size_indiv1}), pos2={pos2}({size_indiv2}), error_dif={error_dif}")
                        print("-----------------------------------------------------")
                pos2 = pos2+1

            elif cambio:
                cambio = False
                pos2 = pos1+1
            else:
                pos1 = pos1+1
                pos2 = pos1+1
                error_dif2 = abs(cost1[pos1]-error_posic)
                if not np.isfinite(error_dif2):
                    error_dif2 = np.Inf
                if error_dif2 >= self.rerank_error:
                    error_posic = cost1[pos1]

        return position
    
    def _selection(self, *args, **kwargs):
        r"""
        Function for selection in GAparsimony.

        Functions implementing selection genetic operator in GA-PARSIMONY after parsimony_rerankprocess. 
        Linear-rank or Nonlinear-rank selection (Michalewicz (1996)). The type of selection is specified 
        with the model selection attribute, it can be: `linear`, `nlinear` or `random`. Modify the attributes: 
        `population`, `fitnessval`, `fitnesstst` and `complexity`.
        """

        # Establezco esta cabecera para permitir una reimplementación si se desea

        if self.selection == "linear":
            r = 2/(self.popSize*(self.popSize-1)) if "r" not in kwargs else kwargs["r"]
            q = 2/self.popSize if "q" not in kwargs else kwargs["q"]

            rank = range(self.popSize)
            prob = map(lambda x: q - (x)*r, rank)

            sel = np.random.choice(list(rank), size=self.popSize, replace=True, p=list(map(lambda x: np.min(np.ma.masked_array(np.array([max(0, x), 1]), np.isnan(np.array([max(0, x), 1])))), prob)))
        
        elif self.selection == "nlinear":
            # Nonlinear-rank selection
            # Michalewicz (1996) Genetic Algorithms + Data Structures = Evolution Programs. p. 60
            q = 0.25 if "q" not in kwargs else kwargs["q"]
            rank = list(range(self.popSize)) # population are sorted
            prob = np.array(list(map(lambda x: q*(1-q)**(x), rank)))
            prob = prob / prob.sum()
            
            sel = np.random.choice(list(rank), size=self.popSize, replace=True, p=list(map(lambda x: np.min(np.ma.masked_array(np.array([max(0, x), 1]), np.isnan(np.array([max(0, x), 1])))), prob)))
        
        elif self.selection == "random":
            sel = np.random.choice(list(range(self.popSize)), size=self.popSize, replace=True)
        
        else:
            raise Exception("Not a valid selection mode provided!!!!!!")
        
        self.population.population = self.population[sel]
        self.fitnessval = self.fitnessval[sel]
        self.fitnesstst = self.fitnesstst[sel]
        self.complexity = self.complexity[sel]
                
    def _crossover(self, parents, alpha=0.1, perc_to_swap=0.5):
        r"""
        Function for crossover in GAparsimony.

        Functions implementing particular crossover genetic operator for GA-PARSIMONY. 
        Method usesfor model parameters Heuristic Blending and random swapping for 
        binary selected features. Modify the attributes: `population`, `fitnessval`, 
        `fitnesstst` and `complexity`.

        Parameters
        ----------
        parents : list
            A list with two integers that correspond to the indices of the rows of the parents 
            of the current population.
        alpha : float, optional
            A tuning parameter for the Heuristic Blending outer bounds [Michalewicz, 1991].
            Typical and default value is `0.1`.
        perc_to_swap : float, optional
            Percentage of features for swapping in the crossovering process. Default value is `0.5`.
        """

        p=parents.copy()

        parents = self.population._pop[parents]
        children = parents # Vector
        pos_param_n = self.population._pos_n
        pos_param_c = self.population._pos_c
        pos_features = np.array(list(range(len(self.population._params), len(self.population._params) + len(self.population.colsnames))))
        
        # Heuristic Blending for parameters not categoricals
        Betas = np.random.uniform(size=len(pos_param_n), low=0, high=1)*(1+2*alpha)-alpha  # 1+alpha*2??????
        children[0,pos_param_n] = parents[0,pos_param_n]-Betas*parents[0,pos_param_n]+Betas*parents[1,pos_param_n]  ## MAP??
        children[1,pos_param_n] = parents[1,pos_param_n]-Betas*parents[1,pos_param_n]+Betas*parents[0,pos_param_n]

        # Random swapping for categorical parameters
        swap_param_c = np.random.uniform(size=len(pos_param_c), low=0, high=1)>=perc_to_swap
        if np.sum(swap_param_c)>0:
    
            parameters_c_parent1 = parents[0,pos_param_c]
            parameters_c_parent2 = parents[1,pos_param_c]
            pos_param_c = np.array(pos_param_c)[swap_param_c]
            children[0,pos_param_c] = parameters_c_parent2[swap_param_c]
            children[1,pos_param_c] = parameters_c_parent1[swap_param_c]
        
        # Random swapping for features
        swap_param = np.random.uniform(size=len(self.population.colsnames), low=0, high=1)>=perc_to_swap
        if np.sum(swap_param)>0:
    
            features_parent1 = parents[0,pos_features]
            features_parent2 = parents[1,pos_features]
            pos_features = pos_features[swap_param]
            children[0,pos_features] = features_parent2[swap_param]
            children[1,pos_features] = features_parent1[swap_param]
  
  
        # correct params that are outside (min and max)
        thereis_min = children[0] < self.population._min
        children[0,thereis_min] = self.population._min[thereis_min]
        thereis_min = children[1] < self.population._min
        children[1,thereis_min] = self.population._min[thereis_min]
        
        thereis_max = children[0] > self.population._max
        children[0,thereis_max] = self.population._max[thereis_max]
        thereis_max = (children[1] > self.population._max)
        children[1,thereis_max] = self.population._max[thereis_max]
  
        aux = np.empty(2)
        aux[:] = np.nan

        self.population[p] = children
        self.fitnessval[p] = aux.copy()
        self.fitnesstst[p] = aux.copy()
        self.complexity[p] = aux.copy()

    #TODO: Update the description (no binary features anymore)
    def _mutation(self):
        r"""
        Function for mutation in GAparsimony.

        Functions implementing mutation genetic operator for GA-PARSIMONY. Method mutes a `GAparsimony.pmutation` percentage 
        of them. If the value corresponds to a model parameter, algorithm uses uniform random mutation. For binary 
        select features, method sets to one if the random value between `[0,1]` is loweror equal to `GAparsimony.feat_mut_thres`. 
        Modify the attributes: `population`, `fitnessval`, `fitnesstst` and `complexity`.
        """


         # Uniform random mutation (except first individual)
        nparam_to_mute = round(self.pmutation*self.population.population.shape[1]*self.popSize)
        if nparam_to_mute<1:
            nparam_to_mute = 1
  
        for _ in range(nparam_to_mute):
            i = np.random.randint((self.not_muted), self.popSize)
            j = np.random.randint(0, self.population.population.shape[1])
            self.population[i,j] = self.population.random_gen[j](j, feat_mut_thres=self.feat_mut_thres)
            self.fitnessval[i] = np.nan
            self.fitnesstst[i] = np.nan
            self.complexity[i] = np.nan

    def _population(self, type_ini_pop="randomLHS"):
        r"""
        Population initialization in GA-PARSIMONY with a combined chromosome of model parameters 
        and selected features. Functions for creating an initial population to be used in the GA-PARSIMONY process.

        Generates a random population of `GAparsimony.popSize` individuals. For each individual a 
        random chromosome is generated with `len(GAparsimony.population._params)` real values in the `range[GAparsimony._min, GAparsimony._max] `
        plus `len(GAparsimony.population.colsnames)` random binary values for feature selection. `random` or Latin Hypercube Sampling can 
        be used to create a efficient spread initial population.

        Parameters
        ----------
        type_ini_pop : list, {'randomLHS', 'geneticLHS', 'improvedLHS', 'maximinLHS', 'optimumLHS'}, optional
            How to create the initial population. `random` optiom initialize a random population between the 
            predefined ranges. Values `randomLHS`, `geneticLHS`, `improvedLHS`, `maximinLHS` & `optimumLHS` 
            corresponds with several meth-ods of the Latin Hypercube Sampling (see `lhs` package for more details).

        Returns
        -------
        numpy.array
            A matrix of dimension `GAparsimony.popSize` rows and `len(GAparsimony.population._params)+len(GAparsimony.population.colsnames)` columns.

        """
  
        nvars = len(self.population._params) + len(self.population.colsnames)
        if type_ini_pop=="randomLHS":
            population = randomLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="geneticLHS":
            population = geneticLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="improvedLHS":
            population = improvedLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="maximinLHS":
            population = maximinLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="optimumLHS":
            population = optimumLHS(self.popSize, nvars, seed=self.seed_ini)
        elif type_ini_pop=="random":
            population = (np.random.rand(self.popSize*nvars) * (nvars - self.popSize) + self.popSize).reshape(self.popSize*nvars, 1)
  
        # Scale matrix with the parameters range
        population = population*(self.population._max-self.population._min)
        population = population+self.population._min
        # Convert features to binary
        #population[:, len(self.population._params):nvars] = population[:, len(self.population._params):nvars]<=self.feat_thres

        return population

    def __str__(self):
        dev =  "An object of class \"ga_parsimony\"\r" + \
                "Available slots:\r" + \
                f"bestfitnessVal: {self.bestfitnessVal}\r" + \
                f"bestfitnessTst: {self.bestfitnessTst}\r" + \
                f"bestcomplexity: {self.bestcomplexity}\r" + \
                f"bestsolution: {self.bestsolution}\r" + \
                f"min_param: {self.population._min}\r" + \
                f"max_param: {self.population._max}\r" + \
                f"nParams: {len(self.population._params)}\r" + \
                f"feat_thres: {self.feat_thres}\r" + \
                f"feat_mut_thres: {self.feat_mut_thres}\r" + \
                f"not_muted: {self.not_muted}\r" + \
                f"rerank_error: {self.rerank_error}\r" + \
                f"iter_start_rerank: {self.iter_start_rerank}\r" + \
                f"nFeatures: {len(self.population.colsnames)}\r" + \
                f"names_param: {list(self.population._params.keys())}\r" + \
                f"names_features: {self.population.colsnames}\r" + \
                f"popSize: {self.popSize}\r" + \
                f"iter: {self.iter}\r" + \
                f"early_stop: {self.early_stop}\r" + \
                f"maxiter: {self.maxiter}\r" + \
                f"minutes_gen: {self.minutes_gen}\r" + \
                f"minutes_total: {self.minutes_total}\r" + \
                f"suggestions: {self.suggestions}\r" + \
                f"population: {self.population}\r" + \
                f"elitism: {self.elitism}\r" + \
                f"pcrossover: {self.pcrossover}\r" + \
                f"pmutation: {self.pmutation}\r" + \
                f"best_score: {self.best_score}\r" + \
                f"solution_best_score: {self.solution_best_score}\r" + \
                f"fitnessval: {self.fitnessval}\r" + \
                f"fitnesstst: {self.fitnesstst}\r" + \
                f"complexity: {self.complexity}\r" + \
                f"summary: {self._summary}\r" + \
                f"bestSolList: {self.bestSolList}\r" + \
                f"history: \r"
        
        for h in self.history:
            dev += (h + "\r")

        return dev

    def __repr__(self):
        return self.__str__()

    def summary(self, **kwargs):
        r"""
        Summary for GA-PARSIMONY.

        Summary method for class `GAparsimony`. If it is assigned, it returns a dict if not displayed on the screen.

        Returns
        -------
        dict
            A `dict` with information about the GAparsimony object.

        """


        x = {"popSize" : self.popSize,
                "maxiter" : self.maxiter,
                "early_stop" : self.early_stop,
                "rerank_error" : self.rerank_error,
                "elitism" : self.elitism,
                "nParams" : len(self.population._params),
                "nFeatures" : len(self.population.colsnames),
                "pcrossover" : self.pcrossover,
                "pmutation" : self.pmutation,
                "feat_thres" : self.feat_thres,
                "feat_mut_thres" : self.feat_mut_thres,
                "not_muted" : self.not_muted,
                "domain" : np.stack([self.population._min, self.population._max], axis=0),
                "suggestions" : self.suggestions,
                "iter" : self.iter,
                "best_score" : self.best_score,
                "bestfitnessVal" : self.bestfitnessVal,
                "bestfitnessTst" : self.bestfitnessTst,
                "bestcomplexity" : self.bestcomplexity,
                "minutes_total" : self.minutes_total,
                "bestsolution" : self.bestsolution,
                "solution_best_score":self.solution_best_score}

        # Para contolar si lo está asignando
        try:
            frame = inspect.currentframe().f_back
            next_opcode = opcode.opname[frame.f_code.co_code[frame.f_lasti+2]]
            if next_opcode not in ["POP_TOP", "PRINT_EXPR"]: # Si no lo asigna
                return x 
        finally:
            del frame 


        print("+------------------------------------+")
        print("|             GA-PARSIMONY           |")
        print("+------------------------------------+\n")
        print("GA-PARSIMONY settings:")
        print(f" Number of Parameters      = {x['nParams']}")
        print(f" Number of Features        = {x['nFeatures']}")
        print(f" Population size           = {x['popSize']}")
        print(f" Maximum of generations    = {x['maxiter']}")
        print(f" Number of early-stop gen. = {x['early_stop']}")
        print(f" Elitism                   = {x['elitism']}")
        print(f" Crossover probability     = {x['pcrossover']}")
        print(f" Mutation probability      = {x['pmutation']}")
        print(f" Max diff(error) to ReRank = {x['rerank_error']}")
        print(f" Perc. of 1s in first popu.= {x['feat_thres']}")
        print(f" Prob. to be 1 in mutation = {x['feat_mut_thres']}")
        
        print("\n Search domain = ")
        print(pd.DataFrame(data=x["domain"], columns=list(self.population._params.keys())+self.population.colsnames, index=["Min_param", "Max_param"]))

        if x["suggestions"] is not None and x["suggestions"].shape[0]>0:
            print("Suggestions =")
            print(x["suggestions"])


        print("\n\nGA-PARSIMONY results:")
        print(f" Iterations                = {x['iter']+1}")
        print(f" Best validation score = {x['best_score']}")
        print(f"\n\nSolution with the best validation score in the whole GA process = \n")
        print(pd.DataFrame(data=x["solution_best_score"][np.newaxis,], columns=["fitnessVal","fitnessTst","complexity"]+list(self.population._params.keys())+self.population.colsnames))
        
        print(f"\n\nResults of the best individual at the last generation = \n")
        print(f" Best indiv's validat.cost = {x['bestfitnessVal']}")
        print(f" Best indiv's testing cost = {x['bestfitnessTst']}")
        print(f" Best indiv's complexity   = {x['bestcomplexity']}")
        print(f" Elapsed time in minutes   = {x['minutes_total']}")
        print(f"\n\nBEST SOLUTION = \n")
        print(pd.DataFrame(data=x["bestsolution"][np.newaxis,], columns=["fitnessVal","fitnessTst","complexity"]+list(self.population._params.keys())+self.population.colsnames))
    

    # Plot a boxplot evolution of val cost, tst cost and complexity for the elitists
    # ------------------------------------------------------------------------------
    def plot(self, min_iter=None, max_iter=None, main_label="Boxplot cost evolution", 
                                steps=5, size_plot=None, *args):
        r"""
        Plot of GA evolution of elitists.

        Plot method shows the evolution of validation and testing errors, and the number 
        of model features selected of elitists. White and grey box-plots represent validation 
        and testing errors of elitists evo-lution, respectively. Continuous and dashed-dotted 
        lines show the validation and testing error ofthe best individual for each generation, 
        respectively. Finally, the shaded area delimits the maximumand minimum number of 
        features, and the dashed line, the number of features of the best individual.

        Parameters
        ----------
        min_iter : int, optional
            Min GA iteration to visualize. Default `None`.
        max_iter : int, optional
            Max GA iteration to visualize.Default `None`.
        main_label : str, optional
            Main plot title.Default 'Boxplot cost evolution'.
        steps : int, optional
            Number of divisions in y-axis. Default `5`.
        size_plot : tuple, optional
            The size of the plot. Default `None`

        """

        if (len(self.history[0])<1):
            print("'GAparsimony.history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
        if not min_iter:
            min_iter = 0
        if not max_iter:
            max_iter = self.iter + 1
        
        # Preparacipon de los datos
        # ==========================
        mat_val = None
        mat_tst = None
        mat_complex = None
        for iter in range(min_iter, max_iter):
            mat_val = np.c_[mat_val, self.history[iter].fitnessval[:self.elitism]] if mat_val is not None else self.history[iter].fitnessval[:self.elitism]
            mat_tst = np.c_[mat_tst, self.history[iter].fitnesstst[:self.elitism]] if mat_tst is not None else self.history[iter].fitnesstst[:self.elitism]

            aux = np.sum(self.history[iter].values[:self.elitism,(len(self.population._params)):(len(self.population._params)+len(self.population.colsnames))], axis=1)
            mat_complex = np.c_[mat_complex, aux] if mat_complex is not None else aux

        x = list(range(min_iter, max_iter))
        mat_val = mat_val.astype(float)
        mat_tst = mat_tst.astype(float)
        mat_complex = mat_complex.astype(float)

        # Grafica
        # ======================
        fig, ax = plt.subplots() if not size_plot else plt.subplots(figsize=size_plot)

        sns.boxplot(x="x", y="y", data=pd.DataFrame(dict(x=np.repeat(range(min_iter, max_iter), self.elitism), y=mat_val.T.flatten())), color=(0.275191, 0.194905, 0.496005), width=0.4)
        sns.boxplot(x="x", y="y", data=pd.DataFrame(dict(x=np.repeat(range(min_iter, max_iter), self.elitism), y=mat_tst.T.flatten())), color=(0.626579, 0.854645, 0.223353), width=0.4)

        plt.suptitle(main_label, fontsize=16)
        plt.title(f"Results for the last best individual: Val={round(self.bestfitnessVal, 5)}, Test={round(self.bestfitnessTst, 5)}, Num.Features={int(mat_complex[0, -1])}")

        # Eje de la derecha
        sns.lineplot(x=x, y=mat_val[0], color=(0.153364, 0.497, 0.557724), style=True, dashes=[(10, 2)])
        ax.legend([],[], frameon=False)

        sns.lineplot(x=x, y=mat_tst[0], color=(0.122312, 0.633153, 0.530398), style=True, dashes=[(2, 2, 10, 2)]) # 2pt line, 2pt break, 10pt line, 2pt break
        ax.legend([],[], frameon=False)
        
        
        ax.set_ylabel("Metric")
        
        # Eje de la izquierda
        ax2 = plt.twinx()
        sns.lineplot(x=x, y=mat_complex[0], color=(0.212395, 0.359683, 0.55171))
        

        plt.fill_between(x = x,
                 y1 = np.min(mat_complex.T, axis=1),
                 y2 = np.max(mat_complex.T, axis=1),
                 alpha = 0.1,
                 facecolor = (0.212395, 0.359683, 0.55171))

        plt.ylabel("Number of Features of Best Indiv.")

        ax.legend(handles=[mlines.Line2D([], [], linestyle="--", color=(0.153364, 0.497, 0.557724), label='Validation METRIC of best individual'),
                            mlines.Line2D([], [], linestyle="-.", color=(0.122312, 0.633153, 0.530398), label='Testing METRIC of best individual'),
                            mlines.Line2D([], [], linestyle="-", color=(0.212395, 0.359683, 0.55171), label='Number of features of best individual'),
                            mpatches.Patch(color=(0.275191, 0.194905, 0.496005), label='Validation metric'),
                            mpatches.Patch(color=(0.626579, 0.854645, 0.223353), label='Testing metric')],
                            loc=3, # 3 o 1
                            ncol=2, borderaxespad=0.)
        
        plt.xticks(np.arange(0,max_iter, steps))
        ax.set_xlabel("Number of Generation")

        ax.set_zorder(ax2.get_zorder()+1) 
        ax.patch.set_visible(False)

        plt.show()        


    def importance(self):
        r"""
        Percentage of appearance of each feature in elitist population.

        Shows the percentage of appearance of each feature in the whole GA-PARSIMONY 
        process butonly for the elitist-population. If it is assigned, it returns a 
        dict if not displayed on the screen.

        Returns
        -------
        numpy.array
            A `numpy.array` with information about feature importance.

        """
    
        if len(self.history[0]) < 1:
            print("'object.history' must be provided!! Set 'keep_history' to TRUE in ga_parsimony() function.")
        min_iter = 1
        max_iter = self.iter
        
        nelitistm = self.elitism
        features_hist = None
        for iter in range(min_iter, max_iter+1):
            data = self.history[iter].iloc[:nelitistm, len(self.population._params):-3].values
            features_hist = data if features_hist is None else np.r_[features_hist, data] ## ANALIZAR CON CUIDADO

        importance = np.mean(features_hist, axis=0)
        sort = order(importance,decreasing = True)
        imp_features = pd.DataFrame(100*importance[order(importance,decreasing = True)][np.newaxis], columns=self.history[0].columns[len(self.population._params):-3][sort])

        # Para contolar si lo está asignando
        try:
            frame = inspect.currentframe().f_back
            next_opcode = opcode.opname[frame.f_code.co_code[frame.f_lasti+2]]
            if next_opcode not in ["POP_TOP", "PRINT_EXPR"]: # Si no lo asigna
                return imp_features 
        finally:
            del frame 

        print("+--------------------------------------------+")
        print("|                  GA-PARSIMONY              |")
        print("+--------------------------------------------+\n")
        print("Percentage of appearance of each feature in elitists: \n")
        print(imp_features)

    def predict(self, X):
        r"""
        Predict result for samples in X.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            Samples.

        Returns
        -------
        numpy.array
            A `numpy.array` with predictions.

        """

        X = X.values if type(X).__name__ == 'DataFrame' else X
        return self.best_model.predict(X[:, self.best_model_conf.columns.astype(bool)])





