"""HYBparsimony for Python is a package for searching accurate parsimonious models by combining feature selection (FS), model
hyperparameter optimization (HO), and parsimonious model selection (PMS) based on a separate cost and complexity evaluation.

To improve the search for parsimony, the hybrid method combines GA mechanisms such as selection, crossover and mutation within a PSO-based optimization algorithm that includes a strategy in which the best position of each particle (thus also the best position of each neighborhood) is calculated taking into account not only the goodness-of-fit, but also the parsimony principle. 

In HYBparsimony, the percentage of variables to be replaced with GA at each iteration $t$ is selected by a decreasing exponential function:
 $pcrossover=max(0.80 \cdot e^{(-\Gamma \cdot t)}, 0.10)$, that is adjusted by a $\Gamma$ parameter (by default $\Gamma$ is set to $0.50$). Thus, in the first iterations parsimony is promoted by GA mechanisms, i.e., replacing by crossover a high percentage of particles at the beginning. Subsequently, optimization with PSO becomes more relevant for the improvement of model accuracy. This differs from other hybrid methods in which the crossover is applied between the best individual position of each particle or other approaches in which the worst particles are also replaced by new particles, but at extreme positions.

Experiments show that, in general, and with a suitable $\Gamma$, HYBparsimony allows to obtain better, more parsimonious and more robust models compared to other methods. It also reduces the number of iterations and, consequently, the computational effort.

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

import copy
import multiprocessing
import random
from multiprocessing import Pool
from functools import partial
from HYBparsimony import Population, order, getFitness
from HYBparsimony.util import parsimony_monitor, parsimony_summary, models
from HYBparsimony.util.fitness import fitness_for_parallel
from HYBparsimony.util.hyb_aux import _rerank, _crossover, _population
from HYBparsimony.lhs import randomLHS
import math
import numpy as np
import pandas as pd
import time
from numpy.random import multinomial
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from HYBparsimony.util.models import check_algorithm

class HYBparsimony(object):

    def __init__(self,
                 fitness = None,
                 features = None,
                 algorithm = None,
                 custom_eval_fun=None,
                 cv=None,
                 scoring=None,
                 type_ini_pop="improvedLHS",
                 npart = 15,
                 maxiter=250,
                 early_stop=None,
                 Lambda=1.0,
                 c1 = 1/2 + math.log(2),
                 c2 = 1/2 + math.log(2),
                 IW_max=0.9,
                 IW_min=0.4,
                 K=3,
                 pmutation = 0.1,
                 #pcrossover_elitists = None,  # an array or a float (number between 0 and 1).
                 #pcrossover = None,  # an array or a float (number between 0 and 1), % of worst individuals to substitute from crossover.
                 gamma_crossover = 0.5,
                 tol = 1e-4,
                 rerank_error=1e-09,
                 keep_history = False,
                 feat_thres = 0.90,
                 best_global_thres = 1,
                 particles_to_delete=None,
                 seed_ini = 1234,
                 not_muted = 3,
                 feat_mut_thres = 0.1,
                 n_jobs=1,
                 verbose=0):
        r"""
            A class for searching parsimonious models by feature selection and parameter tuning with
            an hybrid method based on genetic algorithms and particle swarm optimization.

            Parameters
            ----------
            fitness : function, optional
                The fitness function, any function which takes as input a chromosome which combines the model parameters 
                to tune and the features to be selected. Fitness function returns a numerical vector with three values: validation_cost, 
                testing_cost and model_complexity, and the trained model.
            features : list of str, default=None
                The name of features/columns in the dataset. If None, it extracts the names if X is a dataframe, otherwise it generates a list of the positions according to the value of X.shape[1].
            algorithm: string or dict, default=None
                Id string, the name of the algorithm to optimize (defined in 'HYBparsimony.util.models.py') or a dictionary defined
                with the following properties: {'estimator': any machine learning algorithm compatible with scikit-learn,
                'complexity': the function that measures the complexity of the model, 'the hyperparameters of the algorithm':
                in this case, they can be fixed values (defined by Population.CONSTANT) or a search range $[min, max]$ 
                defined by {"range":(min, max), "type": Population.X} and which type can be of three values: 
                integer (Population.INTEGER), float (Population.FLOAT) or in powers of 10 (Population.POWER), 
                i.e. $10^{[min, max]}$}. If algorithm==None, HYBparsimony uses 'LogisticRegression()' for 
                classification problems, and 'Ridge' for regression problems.
            custom_eval_fun : function, default=None
                An evaluation function similar to scikit-learns's 'cross_val_score()'. If None, HYBparsimony uses
                'cross_val_score(cv=5)'.
            cv: int, cross-validation generator or an iterable, default=None
                Determines the cross-validation splitting strategy (see scikit-learn's 'cross_val_score()' function)
            scoring: str, callable, list, tuple, or dict, default=None.
                Strategy to evaluate the performance of the cross-validated model on the test set. If None cv=5 and 'scoring' is defined as MSE for regression problems, 
                'log_loss' for binary classification problems, and 'f1_macro' for multiclass problems. (see scikit-learn's 
                'cross_val_score()' function)
            type_ini_pop : str, {'randomLHS', 'geneticLHS', 'improvedLHS', 'maximinLHS', 'optimumLHS', 'random'}, optional
                Method to create the first population with `GAparsimony._population` function. Possible values: `randomLHS`, `geneticLHS`, 
                `improvedLHS`, `maximinLHS`, `optimumLHS`, `random`. First 5 methods correspond with several latine hypercube for initial sampling. By default is set to `improvedLHS`.
            npart = int, default=15
                Number of particles in the swarm (population size)
            maxiter = int, default=250
                The maximum number of iterations to run before the HYB process is halted.
            early_stop : int, optional
                The number of consecutive generations without any improvement lower than a difference of 'tol'
                in the 'best_fitness' value before the search process is stopped.
            tol : float, default=1e-4,
                Value defining a significant difference between the 'best_fitness' values between iterations for 'early stopping'.
            rerank_error : float, default=1e-09
                When a value is provided, a second reranking process according to the model complexities is called by `parsimony_rerank` function. 
                Its primary objective isto select individuals with high validation cost while maintaining the robustnessof a parsimonious model. 
                This function switches the position of two models if the first one is more complex than the latter and no significant difference 
                is found between their fitness values in terms of cost. Thus, if the absolute difference between the validation costs are 
                lower than `rerank_error` they are considered similar.
            gamma_crossover : float, default=0.50
                In HYBparsimony, the percentage of variables to be replaced with GA at each iteration $t$ is selected by a decreasing exponential function
                that is adjusted by a 'gamma_crossover' parameter (see references for more info).
            Lambda : float, default=1.0
                PSO parameter (see References)
            c1 : float, default=1/2 + math.log(2)
                PSO parameter (see References)
            c2 : float, default=1/2 + math.log(2)
                PSO parameter (see References)
            IW_max : float, default=0.9
                PSO parameter (see References)
            IW_min : float, default=0.4
                PSO parameter (see References)
            K : int, default=4
                PSO parameter (see References)
            best_global_thres : float, default=1.0
                Percentage of particles that will be influenced by the best global of their neighbourhoods
                (otherwise, they will be influenced by the best of the iteration in each neighbourhood)
                particles_to_delete is not None and len(particles_to_delete) < maxiter:
            particles_to_delete : float, default=None
                The length of the particles to delete is lower than the iterations, 
                the array is completed with zeros up to the number of iterations.
            mutation : float, default=0.1
                The probability of mutation in a parent chromosome. Usually mutation occurs with a small probability. By default is set to `0.10`.
            feat_mut_thres : float, default=0.1
                Probability of the muted `features-chromosome` to be one. Default value is set to `0.10`.
            feat_thres : float, default=0.90
                Proportion of selected features in the initial population. It is recommended a high percentage of the selected features for 
                the first generations.
            keep_history : bool default=False,
                If True keep results of all particles in each iteration into 'history' attribute.
            seed_ini : int, optional
                An integer value containing the random number generator state.
            n_jobs : int, default=1,
                Number of cores to parallelize the evaluation of the swarm. It should be used with caution because the 
                algorithms used or the 'cross_validate()' function used by default to evaluate individuals may also parallelize 
                their internal processes.
            verbose : int, default=0
                The level of messages that we want it to show us. Possible values: 0=silent mode, 1=monitor level,  2=debug level.
        
        Attributes
        ----------
        minutes_total : float
            Total elapsed time (in minutes).
        history : float
            A list with the results of the population of all iterations.'history[iter]' returns a DataFrame 
            with the results of iteration 'iter'.
        best_model
            The best model in the whole optimization process.
        best_score : float
            The validation score of the best model.
        best_complexity : float
            The complexity of the best model.
        selected_features : list,
            The name of the selected features for the best model.
        selected_features_bool : list,
           The selected features for the best model in Boolean form.
        best_model_conf : Chromosome
            The parameters and features of the best model in the whole optimization process.
        
        Examples
        --------
        Usage example for a regression model using the sklearn 'diabetes' dataset 

        .. highlight:: python
        .. code-block:: python

            
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            from sklearn.datasets import load_diabetes
            from sklearn.preprocessing import StandardScaler
            from HYBparsimony import HYBparsimony

            # Load 'diabetes' dataset
            diabetes = load_diabetes()

            X, y = diabetes.data, diabetes.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

            # Standarize X and y
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
            y_test = scaler_y.transform(y_test.reshape(-1,1)).flatten()

            algo = 'KernelRidge'
            HYBparsimony_model = HYBparsimony(algorithm=algo,
                                            features=diabetes.feature_names,
                                            rerank_error=0.001,
                                            verbose=1)

            # Search the best hyperparameters and features 
            # (increasing 'time_limit' to improve RMSE with high consuming algorithms)
            HYBparsimony_model.fit(X_train, y_train, time_limit=0.20)

        .. code-block:: text

            Running iteration 0
            Best model -> Score = -0.510786 Complexity = 9,017,405,352.5 
            Iter = 0 -> MeanVal = -0.88274  ValBest = -0.510786   ComplexBest = 9,017,405,352.5 Time(min) = 0.005858

            Running iteration 1
            Best model -> Score = -0.499005 Complexity = 8,000,032,783.88 
            Iter = 1 -> MeanVal = -0.659969  ValBest = -0.499005   ComplexBest = 8,000,032,783.88 Time(min) = 0.004452

            ...
            ...
            ...

            Running iteration 34
            Best model -> Score = -0.489468 Complexity = 8,000,002,255.68 
            Iter = 34 -> MeanVal = -0.527314  ValBest = -0.489468   ComplexBest = 8,000,002,255.68 Time(min) = 0.007533

            Running iteration 35
            Best model -> Score = -0.489457 Complexity = 8,000,002,199.12 
            Iter = 35 -> MeanVal = -0.526294  ValBest = -0.489457   ComplexBest = 8,000,002,199.12 Time(min) = 0.006522

            Time limit reached. Stopped.

        Usage example for a classification model using the 'breast_cancer' dataset 

        .. highlight:: python
        .. code-block:: python

            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.datasets import load_breast_cancer
            from sklearn.metrics import log_loss
            from HYBparsimony import HYBparsimony
            
            # load 'breast_cancer' dataset
            breast_cancer = load_breast_cancer()
            X, y = breast_cancer.data, breast_cancer.target 
            print(X.shape)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=1)
            
            # Standarize X and y (some algorithms require that)
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)

            HYBparsimony_model = HYBparsimony(features=breast_cancer.feature_names,
                                            rerank_error=0.005,
                                            verbose=1)
            HYBparsimony_model.fit(X_train, y_train, time_limit=0.50)
            # Extract probs of class==1
            preds = HYBparsimony_model.predict_proba(X_test)[:,1]
            print(f'\n\nBest Model = {HYBparsimony_model.best_model}')
            print(f'Selected features:{HYBparsimony_model.selected_features}')
            print(f'Complexity = {round(HYBparsimony_model.best_complexity, 2):,}')
            print(f'5-CV logloss = {-round(HYBparsimony_model.best_score,6)}')
            print(f'logloss test = {round(log_loss(y_test, preds),6)}')
        
        .. code-block:: text

            (569, 30)
            Detected a binary-class problem. Using 'neg_log_loss' as default scoring function.
            Running iteration 0
            Best model -> Score = -0.091519 Complexity = 29,000,000,005.11 
            Iter = 0 -> MeanVal = -0.297448  ValBest = -0.091519   ComplexBest = 29,000,000,005.11 Time(min) = 0.006501

            Running iteration 1
            Best model -> Score = -0.085673 Complexity = 27,000,000,009.97 
            Iter = 1 -> MeanVal = -0.117216  ValBest = -0.085673   ComplexBest = 27,000,000,009.97 Time(min) = 0.004273

            ...
            ...

            Running iteration 102
            Best model -> Score = -0.064557 Complexity = 11,000,000,039.47 
            Iter = 102 -> MeanVal = -0.076314  ValBest = -0.066261   ComplexBest = 9,000,000,047.25 Time(min) = 0.004769

            Running iteration 103
            Best model -> Score = -0.064557 Complexity = 11,000,000,039.47 
            Iter = 103 -> MeanVal = -0.086243  ValBest = -0.064995   ComplexBest = 11,000,000,031.2 Time(min) = 0.004591

            Time limit reached. Stopped.

            Best Model = LogisticRegression(C=5.92705799354935)
            Selected features:['mean texture' 'mean concave points' 'radius error' 'area error'
            'compactness error' 'worst radius' 'worst perimeter' 'worst area'
            'worst smoothness' 'worst concavity' 'worst symmetry']
            Complexity = 11,000,000,039.47
            5-CV logloss = 0.064557
            logloss test = 0.076254
                
        """

        self.type_ini_pop = type_ini_pop
        self.fitness = fitness
        self.features = features
        self.npart = npart
        self.maxiter = maxiter
        self.early_stop = maxiter if not early_stop else early_stop
        self.Lambda = Lambda
        self.c1 = c1
        self.c2 = c2
        self.IW_max = IW_max
        self.IW_min = IW_min
        self.K = K
        self.tol = tol

        self.rerank_error = rerank_error
        self.verbose = verbose
        self.seed_ini = seed_ini

        if pmutation is None:
            self.pmutation = 0.0
        else:
            self.pmutation = pmutation
        self.not_muted = not_muted
        self.feat_mut_thres = feat_mut_thres

        self.feat_thres = feat_thres

        self.minutes_total = 0
        self.history = list()
        self.keep_history = keep_history

        # Percentage of particles that will be influenced by the best global of their neighbourhoods
        # (otherwise, they will be influenced by the best of the iteration in each neighbourhood)
        self.best_global_thres = best_global_thres

        # if pcrossover is not None:
        #     if isinstance(pcrossover,(list,np.ndarray)): #If it is a list or an np array
        #         if len(pcrossover) < maxiter:
        #             # If the length of the pcrossover array is lower than the iterations, the array is completed with zeros
        #             # up to the number of iterations.
        #             self.pcrossover = np.zeros(maxiter).astype(float)
        #             self.pcrossover[:len(pcrossover)] = pcrossover[:]
        #         else:
        #             self.pcrossover = pcrossover
        #     else:
        #         # If the parameter was a float, then an array is built in which each position contains that float.
        #         self.pcrossover = np.full(maxiter, pcrossover, dtype=float)
        #     # Ensure all numbers are in the range [0,1]
        #     self.pcrossover[self.pcrossover < 0] = 0
        #     self.pcrossover[self.pcrossover > 1] = 1
        # else:
        #     self.pcrossover = None

        # El gamma del crossover (ahora construyo el self.pcrossover a partir del gamma).
        self.pcrossover = None
        if gamma_crossover != 0.0:
            perc_malos = 0.80 * np.exp(-gamma_crossover * np.arange(self.maxiter))
            perc_malos[perc_malos < 0.10] = 0.10
            self.pcrossover = perc_malos

        self.n_jobs=n_jobs
        if self.n_jobs < 1:
            self.n_jobs = multiprocessing.cpu_count()  # Si ponemos un -1, entonces todos los cores (aunque la validación cruzada ya hará más aún!).

        if particles_to_delete is not None and len(particles_to_delete) < maxiter:
            # If the length of the particles to delete is lower than the iterations, the array is completed with zeros
            # up to the number of iterations.
            self.particles_to_delete = np.zeros(maxiter).astype(int)
            self.particles_to_delete[:len(particles_to_delete)] = particles_to_delete[:]
        else:
            self.particles_to_delete = particles_to_delete

        if self.seed_ini:
            np.random.seed(self.seed_ini)

        # Custom cross val score
        self.custom_eval_fun = custom_eval_fun

        self._cv=cv
        self._scoring=scoring
        self.algorithm = algorithm


    def fit(self, X, y, time_limit=None):
        r"""
        Performs the search of accurate parsimonious models by combining feature selection, hyperparameter optimizacion,
            and parsimonious model selection (PMS) with data matrix (X) and targets (y).

        Parameters
        ----------
        X : pandas.DataFrame or numpy.array
            Training vector.
        y : pandas.DataFrame or numpy.array
            Target vector relative to X.
        time_limit : float, default=None
            Maximum time to perform the optimization process in minutes.
        """

        #############################################
        #  SOME LOGIC ON PARAMETERS' INITIALIZATION
        #############################################

        # Detect type of problem and define default scoring function.
        def check_classification(y):
           return np.issubdtype(y.dtype, np.integer)
        
        if self._scoring is not None:
            default_scoring = self._scoring
            if self.verbose > 0:
                print(f"Using '{default_scoring}' as scoring function.")
        elif check_classification(y):
            if len(np.unique(y))==2:
                default_scoring = 'neg_log_loss'
                if self.verbose > 0:
                    print("Detected a binary-class problem. Using 'neg_log_loss' as default scoring function.")
            else:
                default_scoring = 'f1_macro'
                if self.verbose > 0:
                    print("Detected a multi-class problem. Using 'f1_macro' as default scoring function.")
        else:
            default_scoring = 'neg_mean_squared_error'
            if self.verbose > 0:
                print("Detected a regression problem. Using 'neg_mean_squared_error' as default scoring function.")

        def default_cv_score(estimator, X, y):
            return cross_val_score(estimator, X, y, cv=5, scoring=default_scoring)

        # Create custom_eval_fun 
        if self._cv is not None and self.custom_eval_fun is None:
            if self._scoring is not None:
                self.custom_eval_fun = partial(cross_val_score, cv=self._cv, scoring=self._scoring)
            else: # Por defecto:
                self.custom_eval_fun = partial(cross_val_score, cv=self._cv, scoring=default_scoring)
        elif self.custom_eval_fun is None: 
            if self._scoring is not None: 
                self.custom_eval_fun = partial(cross_val_score, scoring=self._scoring)
            else:
                self.custom_eval_fun = default_cv_score

        # Select and check algorithm from dictionary
        self.algorithm = check_algorithm(self.algorithm, check_classification(y))
        self.params = {k: self.algorithm[k] for k in self.algorithm.keys() if k not in ["estimator", "complexity"]}

        # Fitness function
        if self.n_jobs == 1:
            self.fitness = getFitness(self.algorithm['estimator'], self.algorithm['complexity'],
                                      self.custom_eval_fun)
        else: # Parallelization
            self.fitness = partial(fitness_for_parallel, self.algorithm['estimator'],
                                   self.algorithm['complexity'], self.custom_eval_fun)

        if self.n_jobs > 1:
            pool = Pool(self.n_jobs)

        if self.features is None: # Si no hay features (nombre de las columnas a optimizar), entonces cojo todas
            if "pandas" in str(type(X)):
                self.features = X.columns # Si es un DataFrame, saco los nombres de las columnas.
            else: # SI no, entonces es un numpy array y pongo números del 0 al número de columnas
                num_rows, num_cols = X.shape
                self.features = list(range(num_cols))

        #############################################
        #               THE HYBRID METHOD
        #############################################

        start_time = time.time()

        if self.seed_ini:
            np.random.seed(self.seed_ini)

        population = Population(self.params, columns=self.features)
        population.population = _population(population, seed_ini=self.seed_ini, popSize=self.npart,
                                            type_ini_pop=self.type_ini_pop)  # To create the initial population


        # Update population to satisfy the feat_thres
        population.update_to_feat_thres(self.npart, self.feat_thres)

        nfs = len(population.colsnames)
        nparams = len(population._params)
        self._summary = np.empty((self.maxiter, 6 * 2,))
        self._summary[:] = np.nan
        self.best_score = np.NINF
        self.best_complexity = np.Inf

        maxFitness = np.Inf
        best_fit_particle = np.empty(self.npart)
        best_fit_particle[:] = np.NINF

        best_pos_particle = np.empty(shape=(self.npart, nparams + nfs))
        best_complexity_particle = np.empty(self.npart)  # Complexities
        best_complexity_particle[:] = np.Inf

        range_numbers = population._max - population._min
        vmax = self.Lambda * range_numbers
        range_as_pd = pd.Series(range_numbers)
        lower_as_pd = pd.Series(population._min)
        v_norm = randomLHS(self.npart, nparams + nfs)
        v_norm = pd.DataFrame(v_norm)
        v_norm = v_norm.apply(lambda row: row * range_as_pd, axis=1)
        v_norm = v_norm.apply(lambda row: row + lower_as_pd, axis=1)

        velocity = (v_norm - population._pop) / 2
        velocity = velocity.to_numpy()

        self.bestSolList = list()
        self.best_models_list = list()
        self.best_models_conf_list = list()

        # Variables to store the best global positions, fitnessval and complexity of each particle
        bestGlobalPopulation = copy.deepcopy(population._pop)
        bestGlobalFitnessVal = np.empty(self.npart)
        bestGlobalFitnessVal[:] = np.NINF
        bestGlobalComplexity = np.empty(self.npart)
        bestGlobalComplexity[:] = np.inf

        #Variable that tracks the deleted particles (their number in the table)
        deleted_particles = []
        valid_particles = [x for x in range(self.npart) if
                           x not in deleted_particles]  # valid particles (numbers in the table)

        fitnessval = np.empty(self.npart)
        fitnessval[:] = np.nan
        fitnesstst = np.empty(self.npart)
        fitnesstst[:] = np.nan
        complexity = np.empty(self.npart)
        complexity[:] = np.nan
        _models = np.empty(self.npart).astype(object)
        _models[:] = None

        update_neighbourhoods = False
        crossover_applied = False

        for iter in range(self.maxiter):
            if self.verbose > 0:
                print("Running iteration", iter)
            
            tic = time.time()
            #####################################################
            # Compute solutions
            #####################################################

            if self.n_jobs == 1: # Si NO hay paralelismo (comportamiento por defecto)
                for t in valid_particles:
                    c = population.getChromosome(t)

                    if np.sum(c.columns) > 0:
                        fit = self.fitness(c, X=X, y=y)
                        fitnessval[t] = fit[0][0]
                       # fitnesstst[t] = fit[0][1]
                        complexity[t] = fit[0][1]
                        _models[t] = fit[1]

            else:
                list_params = []
                for t in valid_particles:  # Se entrenan todas siempre (salvo las que eliminemos del proceso)
                    c = population.getChromosome(t)
                    if np.sum(c.columns) > 0:
                        list_params.append([c,X,y])

                results = pool.starmap(self.fitness, list_params)  ## Aquí se hace el paralelismo.
                # Recorremos los resultados
                for fit, t in zip(results, valid_particles):
                    fitnessval[t] = fit[0][0]
                    #fitnesstst[t] = fit[0][1]
                    complexity[t] = fit[0][1]
                    _models[t] = fit[1]

            if self.seed_ini:
                np.random.seed(self.seed_ini * iter)

            # Sort by the Fitness Value
            # ----------------------------
            sort = order(fitnessval, kind='heapsort', decreasing=True, na_last=True)

            PopSorted = population[sort, :].copy()
            FitnessValSorted = fitnessval[sort]
            #FitnessTstSorted = fitnesstst[sort]
            ComplexitySorted = complexity[sort]
            _modelsSorted = _models[sort]

            if self.verbose == 2:
                print("\nStep 1. Fitness sorted")
                print(np.c_[FitnessValSorted, ComplexitySorted, population.population][:10, :])
                # input("Press [enter] to continue")

            if self.rerank_error != 0.0:
                ord_rerank = _rerank(FitnessValSorted, ComplexitySorted, self.npart, self.rerank_error)
                PopSorted = PopSorted[ord_rerank]
                FitnessValSorted = FitnessValSorted[ord_rerank]
               # FitnessTstSorted = FitnessTstSorted[ord_rerank]
                ComplexitySorted = ComplexitySorted[ord_rerank]
                _modelsSorted = _modelsSorted[ord_rerank]

                if self.verbose == 2:
                    print("\nStep 2. Fitness reranked")
                    print(np.c_[FitnessValSorted, ComplexitySorted, population.population][:10, :])
                    # input("Press [enter] to continue")


            # Keep results
            # ---------------
            self._summary[iter, :] = parsimony_summary(FitnessValSorted, ComplexitySorted)

            # Keep Best Solution of this iteration
            # ------------------
            bestfitnessVal = FitnessValSorted[0]
            #bestfitnessTst = FitnessTstSorted[0]
            bestcomplexity = ComplexitySorted[0]
            bestIterSolution = np.concatenate([[bestfitnessVal, bestcomplexity], PopSorted[0]])
            self.bestSolList.append(bestIterSolution)
            self.best_models_list.append(_modelsSorted[0])
            self.best_models_conf_list.append(PopSorted[0])

            # Keep Global Best Model
            # ------------------
            # The best_score of the whole process. It is update if we find a better score, or equal but with lower complexity.
            if bestfitnessVal > self.best_score or (bestfitnessVal == self.best_score and bestcomplexity < self.best_complexity):
                self.best_score = bestfitnessVal
                self.best_complexity = bestcomplexity
                self.bestsolution = bestIterSolution
                self.solution_best_score = np.r_[self.best_score, bestfitnessVal, bestcomplexity]
                self.best_model = _modelsSorted[0]
                self.best_model_conf = PopSorted[0].copy()
                # print("ACTUALIZO", self.best_model.C, self.best_model_conf)
                # if self.best_model_conf[0] != self.best_model.C:
                #     print("problemas")
                #     print("MODELS", _modelsSorted)
                #     print("POPSORTED", PopSorted)
                #     print("fitnessvalsorted", FitnessValSorted)
            # if self.verbose > 0:
            #     print("Current best score:", self.best_score)
                

            # Update global best positions, fitness and complexity of each particle (with NO rerank)
            for i in range(self.npart):
                if fitnessval[i] > bestGlobalFitnessVal[i] or (fitnessval[i] == bestGlobalFitnessVal[i] and complexity[i] < bestGlobalComplexity[i]):
                    bestGlobalPopulation[i,:] = population._pop[i,:]
                    bestGlobalFitnessVal[i] = fitnessval[i]
                    bestGlobalComplexity[i] = complexity[i]


            # Keep elapsed time in minutes
            # ----------------------------
            tac = time.time()
            elapsed_gen = (tac - tic) / 60.0
            self.minutes_total += + elapsed_gen

            # Keep this generation into the History list (with no order)
            # ------------------------------------------
            if self.keep_history:
                self.history.append(
                    pd.DataFrame(np.c_[population.population, fitnessval, fitnesstst, complexity],
                                 columns=list(population._params.keys()) + population.colsnames + ["fitnessval", "fitnesstst",
                                                                                                   "complexity"]))


            # Call to 'monitor' function
            # --------------------------
            if self.verbose > 0:
                parsimony_monitor(iter, self.best_score, self.best_complexity, 
                                  fitnessval, bestfitnessVal, bestcomplexity, elapsed_gen)

            if self.verbose == 2:
                print("\nStep 3. Fitness results")
                print(np.c_[FitnessValSorted, ComplexitySorted, population.population][:10, :])
                # input("Press [enter] to continue")

            #print((population._pop))
            #print((population._pop[sort])[ord_rerank])
            #print((fitnessval[sort])[ord_rerank])

            # Exit?
            # -----
            best_val_cost = self._summary[:, 0][~np.isnan(self._summary[:, 0])]
            if bestfitnessVal >= maxFitness:
                break
            if iter == self.maxiter:
                break
            if (len(best_val_cost) - (np.min(np.arange(len(best_val_cost))[best_val_cost >= (np.max(best_val_cost) - self.tol)]))) >= self.early_stop:
                if self.verbose > 0:
                    print("Early stopping reached. Stopped.")
                break
            if time_limit is not None and time_limit < (time.time() - start_time)/60:
                if self.verbose > 0:
                    print("Time limit reached. Stopped.")
                break

            ####################################################
            # Deletion step (disabled by default)
            ####################################################
            if self.particles_to_delete is not None and self.particles_to_delete[iter]>0:
                # particles_to_delete[iter] contains the number of particles to be deleted in that iteration
                # We delete the worse particles at that point (in global, not in that iteration).
                sort1 = order(bestGlobalFitnessVal, kind='heapsort', decreasing=True, na_last=True)
                sort_not_deleted = [x for x in sort1 if x not in deleted_particles]
                deleted_particles = deleted_particles + sort_not_deleted[-self.particles_to_delete[iter]:]
                valid_particles = [x for x in range(self.npart) if x not in deleted_particles]
                update_neighbourhoods = True




            #####################################################
            # Generation of the Neighbourhoods
            #####################################################
            # If there is no improvement in the current iteration, the neighbourhood is changed. It also changes if particles have been deleted.
            if FitnessValSorted[0] <= self.best_score or update_neighbourhoods:
                update_neighbourhoods = False
                nb = list()
                for i in range(self.npart):
                    # Each particle informs at random K particles (the same particle may be chosen several times), and informs itself.
                    # The parameter K is usually set to 3. It means that each particle informs at less one particle (itself), and at most K+1 particles (including itself)

                    # Thus, a random integer vector of K elements between 0 and npart-1 is created and we append the particle.
                    # Duplicates are removed and this represents the neighbourhood.
                    if i not in deleted_particles:
                        #nb.append(np.unique(np.append(np.random.randint(low=0, high=self.npart - 1, size=self.K), i)))

                        indices = np.random.randint(low=0, high=len(valid_particles), size=self.K) # High is not included
                        random_particles = [valid_particles[index] for index in indices]
                        nb.append(np.unique(np.append(random_particles, i)))
                    else:
                        nb.append(np.nan)

                # Create an array to decide if a particle must be influenced by the best global of the neighbourhoods or the best of the iteration
                nb_global = np.random.choice(a=[True, False], size=(self.npart,), p=[self.best_global_thres, 1-self.best_global_thres])



            ###########################################
            # Update particular global bests (best position of the particle in the whole process, wrt to rerank)
            ###########################################

            # We have to take care to avoid problems with rerank:
            # EXAMPLE (rerank = 0.08):
            # SCORE 0.80 0.85 0.90
            # COST    10  100  200
            # The best score wrt to rerank should be 0.85. But if we get 0.80 with cost 10 in the next
            # iteration, that would be chosen. This is wrong, since we would be moving to worse scores. The
            # rerank must be applied wrt the best global score of each particle.
            for t in [p for p in range(self.npart) if np.isfinite(fitnessval[p])]:# Solo cogemos las partículas que tienen fitnessval finito (que no sea Nan ni inf)
                # Three cases:
                # (1) If the best improves much, then update.
                # (2) If the best does not improve much, but the complexity is lower, then update.
                # (3) Otherwise, rerank criterion, but "consuming the rerank" wrt to the global best.
                if (fitnessval[t] > best_fit_particle[t] + self.rerank_error) \
                    or (fitnessval[t] >= best_fit_particle[t] and complexity[t] < best_complexity_particle[t]) \
                    or (best_fit_particle[t] - fitnessval[t]) <= self.rerank_error - (bestGlobalFitnessVal[t] - best_fit_particle[t]) and complexity[t] < best_complexity_particle[t]:
                    best_fit_particle[t] = fitnessval[t]  # Update the particular best fit of that particle.
                    best_pos_particle[t, :] = population._pop[t, :]  # Update the particular best pos of that particle.
                    best_complexity_particle[t] = complexity[t] # Update the complexity (could be more complex if the fitnessval[t] is better)

            ###########################################
            # Compute Local bests in the Neighbourhoods
            ###########################################
            best_pos_neighbourhood = np.empty(shape=(self.npart, nparams + nfs))  # Matrix in which i-th row contains the best particle of the i-th neighbourhood.
            best_fit_neighbourhood = np.empty(self.npart)  # Array that contains in position i the score of the best particle of the i-th neighbourhood.
            best_fit_neighbourhood[:] = np.Inf

            for i in valid_particles:

                if nb_global[i]: # If the global best of the neighbourhood must be selected
                    particles_positions = nb[i]  # Positions of the neighbourhood particles (number within population)
                    local_fits = best_fit_particle[particles_positions]
                    local_complexity = best_complexity_particle[particles_positions]
                    local_sort = order(local_fits, kind='heapsort', decreasing=True, na_last=True)
                    local_fits_sorted = local_fits[local_sort]
                    local_complexity_sorted = local_complexity[local_sort]
                    local_sort_rerank = _rerank(local_fits_sorted, local_complexity_sorted, len(local_fits),
                                                self.rerank_error, preserve_best=True)
                    max_local_fit_pos = particles_positions[local_sort[local_sort_rerank[0]]]
                    best_pos_neighbourhood[i, :] = best_pos_particle[max_local_fit_pos, :]
                    #best_fit_neighbourhood[i] = best_fit_particle[max_local_fit_pos]

                else: # The best of the neighbourhood in the current iteration
                    particles_positions = nb[i]  # Positions of the neighbourhood particles (number within population)
                    local_fits = fitnessval[particles_positions]

                    local_complexity = complexity[particles_positions]
                    local_sort = order(local_fits, kind='heapsort', decreasing=True, na_last=True)
                    local_fits_sorted = local_fits[local_sort]
                    local_complexity_sorted = local_complexity[local_sort]
                    local_sort_rerank = _rerank(local_fits_sorted,local_complexity_sorted, len(local_fits), self.rerank_error, preserve_best=False)
                    max_local_fit_pos = particles_positions[local_sort[local_sort_rerank[0]]]

                    best_pos_neighbourhood[i, :] = population._pop[max_local_fit_pos, :]
                    #best_fit_neighbourhood[i] = fitnessval[max_local_fit_pos]


            ######################
            # Crossover step
            ######################

            indexes_worst_particles = []
            if self.pcrossover is not None and self.pcrossover[iter] > 0:
                ######################
                # Selection substep
                ######################
                # Nonlinear-rank selection
                # Michalewicz (1996) Genetic Algorithms + Data Structures = Evolution Programs. p. 60
                q = 0.25
                rank = list(range(self.npart))
                prob = np.array(list(map(lambda x: q * (1 - q) ** (x), rank)))
                prob = prob / prob.sum() # En prob, metemos las probabilidades. El primer elemento tiene más probabilidad, y así sucesivamente.
                # Ahora en sel, aplicamos esas probabilidades para seleccionar, teniendo en cuenta que los índices de las mejores están en sort[ord_rerank]
                # (porque la población no está ordenada, así que no podemos usar rank como en GA).
                sel = np.random.choice(sort[ord_rerank], size=self.npart, replace=True, p=list(map(lambda x: np.min(
                    np.ma.masked_array(np.array([max(0, x), 1]), np.isnan(np.array([max(0, x), 1])))), prob)))
                # Cambia la población para seleccionar los que se van a reproducir. Puede haber filas repetidas en population.
                # Así, luego se pueden cruzar más veces.
                population_selection = copy.deepcopy(population) # Hago deepcopy porque es array de arrays.
                population_selection._pop = population_selection._pop[sel]
                fitnessval_selection = fitnessval[sel].copy()
                #fitnesstst_selection = fitnesstst[sel].copy()
                complexity_selection = complexity[sel].copy()
                velocity_selection = velocity[sel].copy()

                ######################
                # Crossover substep
                ######################

                nmating = int(np.floor(self.npart / 2))
                mating = np.random.choice(list(range(2 * nmating)), size=(2 * nmating), replace=False).reshape((nmating, 2))

                # Hacemos crossover de la población seleccionada
                population_crossover = copy.deepcopy(population_selection)
                fitnessval_crossover = fitnessval_selection.copy()
                #fitnesstst_crossover = fitnesstst_selection.copy()
                complexity_crossover = complexity_selection.copy()
                velocity_crossover = velocity_selection.copy()

                for i in range(nmating):
                    parents_indexes = mating[i,]
                    # Voy haciendo el crossover en la nueva población
                    _crossover(population_crossover, velocity_crossover, fitnessval_crossover, complexity_crossover,
                               parents_indexes, children_indexes=parents_indexes)

                # Ahora cojo la población original, y sustituyo el % de malos a sustituir por individuos aleatorios de la población del crossover.
                npart_worst = max(1, int(np.floor(self.npart * self.pcrossover[iter])))
                indexes_worst_particles = sort[ord_rerank[-npart_worst:]]
                # Array aleatorio de tamaño npart y números entre 0 y npart - 1. También podría hacer un suffle.
                # No repito aquí (pero podrá haber padres repetidos porque en population_crossover podría haber filas repetidas):
                random_array = np.random.choice(range(self.npart), self.npart, replace=False)
                for i in indexes_worst_particles: #Esto ya me asegura que no toco los elitistas, solo sustituyo las partículas malas.
                    population._pop[i] = population_crossover._pop[random_array[i]]
                    fitnessval[i] = fitnessval_crossover[random_array[i]]
                    #fitnesstst[i] = fitnesstst_crossover[random_array[i]]
                    complexity[i] = complexity_crossover[random_array[i]]
                    velocity[i] = velocity[random_array[i]]

            #####################################################
            # Update positions and velocities following SPSO 2007
            #####################################################

            # Solo tengo que actualizar los que no haya sustituido.
            indexes_except_substituted_particles = [i for i in range(self.npart) if i not in indexes_worst_particles]

            U1 = np.random.uniform(low=0, high=1,
                                   size=(self.npart, nparams + nfs))  # En el artículo se llaman r1 y r2
            U2 = np.random.uniform(low=0, high=1,
                                   size=(self.npart, nparams + nfs))  # En el artículo se llaman r1 y r2

            IW = self.IW_max - (self.IW_max - self.IW_min) * iter / self.maxiter

            # Two first terms of the velocity

            velocity[indexes_except_substituted_particles,:] = IW * velocity[indexes_except_substituted_particles,:] \
                                                         + U1[indexes_except_substituted_particles,:] * self.c1 * (best_pos_particle[indexes_except_substituted_particles,:] - population._pop[indexes_except_substituted_particles,:])

            velocity[indexes_except_substituted_particles,:] = velocity[indexes_except_substituted_particles,:] + self.c2 * U2[indexes_except_substituted_particles,:] * (
                        best_pos_neighbourhood[indexes_except_substituted_particles,:] - population._pop[indexes_except_substituted_particles,:])

            # Limit velocity to vmax to avoid explosion

            for j in range(nparams + nfs):
                vmax_pos = np.where(abs(velocity[:,j]) > vmax[j])[0]
                for i in vmax_pos:
                    velocity[i, j] = math.copysign(1, velocity[i, j]) * abs(vmax[j])

            ##############################
            # Update positions of FEATURES
            ##############################

            for nf in range(nparams,nparams + nfs): # We must move to the features (the particles contain first hyper-parameters and then features)
                for p in indexes_except_substituted_particles:
                    population._pop[p,nf] = population._pop[p,nf] + velocity[p,nf] # Update positions for the model positions (x = x + v)
                    # To ensure that the interval [0,1] is preserved
                    if population._pop[p, nf] > 1.0:
                        population._pop[p, nf] = 1.0
                    if population._pop[p,nf] < 0.0:
                        population._pop[p, nf] = 0.0


            ######################
            # Mutation of FEATURES
            # ####################

            if self.pmutation > 0:
                # Uniform random mutation (except first individual)
                nfts_to_mute = round(self.pmutation * nfs * self.npart)
                if nfts_to_mute < 1:
                    nfts_to_mute = 1
                indexes_to_mute = sort[ord_rerank[self.not_muted:]]
                for _ in range(nfts_to_mute):
                    i = np.random.choice(indexes_to_mute)
                    j = np.random.randint(0, nfs - 1)
                    population._pop[i, nparams + j] = population.random_gen[j](j, feat_mut_thres=self.feat_mut_thres)
                    fitnessval[i] = np.nan
                    fitnesstst[i] = np.nan
                    complexity[i] = np.nan


            # if self.pmutation > 0:
            #     rnd_mut = np.random.uniform(size = (self.npart, nfs))
            #     for p in range(self.npart):
            #         for nf in range(nparams,nparams + nfs):
            #             if rnd_mut[p, nf - nparams] < self.pmutation:
            #                 if population._pop[p, nf] < 0.5:
            #                     population._pop[p, nf] = np.random.uniform(low=0.5, high=1.0)
            #                 else:
            #                     population._pop[p, nf] = np.random.uniform(low=0.0, high=0.5)


            #######################################################
            # Update positions of model HYPERPARAMETERS (x = x + v)
            #######################################################

            for j in range(nparams):
                population._pop[indexes_except_substituted_particles, j] = \
                    population._pop[indexes_except_substituted_particles, j] + velocity[indexes_except_substituted_particles, j]

            ################################################################################################
            # Confinement Method for SPSO 2007 - absorbing2007 (hydroPSO) - Deterministic Back (Clerc, 2007)
            ################################################################################################
            for j in range(nparams):
                out_max = (population._pop[:, j] > population._max[j])
                out_min = (population._pop[:, j] < population._min[j])
                population._pop[out_max, j] = population._max[j]
                population._pop[out_min, j] = population._min[j]
                velocity[out_max, j] = 0
                velocity[out_min, j] = 0

            # ASEGURARNOS QUE AL MENOS UNA FEATURE SE SELECCIONA EN CADA PARTICULA
            # TODO: Esto debería hacerse en otro lado!
            for i in range(self.npart):  # the particles contain first hyper-parameters and then feature
                aux = population._pop[i, nparams:]
                if (aux<0.5).all():
                    feature_to_change = random.randint(nparams, nparams + nfs - 1)
                    new_value = random.uniform(0.5, 1)
                    population._pop[i, feature_to_change] = new_value


        if self.n_jobs>1:
            pool.close()

        # Guardo las features seleccionadas
        aux = self.best_model_conf[nparams:nparams + nfs]
        self.selected_features_boolean = (aux >= 0.5) # Me guardo como una lista de booleanos si las features están o no
        self.selected_features = np.array(self.features)[self.selected_features_boolean] # Me guardo los nombres
        if self.verbose == 2:
            print("Selected features:", self.selected_features)
        return self.best_model

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
        num_rows, num_cols = X.shape
        if num_cols == len(self.selected_features): #Si nos han pasado un X donde ya he cogido las columnas que debía coger
            preds = self.best_model.predict(X)
        else: # En otro caso, nos han pasado un X entero y nos tenemos que quedar solo con las columnas seleccionadas.
            if isinstance(X, pd.DataFrame): # Si es un dataframe, puedo acceder a las columnas por nombre/booleano
                X_selected_features = X[self.selected_features].values
            else: #Si es un Numpy, entonces tengo que quedarme con las columnas apropiadas
                X_selected_features = X[:,self.selected_features_boolean] # Cojo todas las filas pero solo las columnas apropiadas.
            preds = self.best_model.predict(X_selected_features)
        return preds


    def predict_proba(self, X):
        r"""
        Predict probabilities for each class and sample in X (only for classification models).

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            Samples.

        Returns
        -------
        numpy.array
            A `numpy.array` with predictions. Returns the probability of the sample for each class in the model.

        """
        num_rows, num_cols = X.shape
        if num_cols == len(self.selected_features): #Si nos han pasado un X donde ya he cogido las columnas que debía coger
            preds = self.best_model.predict_proba(X)
        else: # En otro caso, nos han pasado un X entero y nos tenemos que quedar solo con las columnas seleccionadas.
            if isinstance(X, pd.DataFrame): # Si es un dataframe, puedo acceder a las columnas por nombre/booleano
                X_selected_features = X[self.selected_features].values
            else: #Si es un Numpy, entonces tengo que quedarme con las columnas apropiadas
                X_selected_features = X[:,self.selected_features_boolean] # Cojo todas las filas pero solo las columnas apropiadas.
            preds = self.best_model.predict_proba(X_selected_features)
        return preds

