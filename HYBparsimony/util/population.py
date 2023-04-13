# -*- coding: utf-8 -*-

import numpy as np

class Population:

    INTEGER = 0
    FLOAT = 1
    CATEGORICAL = 2
    CONSTANT = 3
    POWER = 4

    def __init__(self, params, columns, population = None):
        r"""
        This class is used to create the GA populations. 
        Allow chromosomes to have int, float, and constant values. 


        Parameters
        ----------
        params : dict
            It is a dictionary with the model's hyperparameters to be adjusted and the search space of them.
            
            .. code-block::

                {
                    "<< hyperparameter name >>": {
                        "range": [<< minimum value >>, << maximum value >>],
                        "type": GAparsimony.FLOAT/GAparsimony.INTEGER
                    },
                    "<< hyperparameter name >>": {
                        "value": << constant value >>,
                        "type": GAparsimony.CONSTANT
                    }
                }

        columns : int or list of str
            The number of features/columns in the dataset or a list with their names.
        population : numpy.array, optional
            It is a float matrix that represents the population. Default `None`.

        Attributes
        ----------
        population : Population
            The population.
        _min : numpy.array
            A vector of length `params+columns` with the smallest values that can take.
        _max : numpy.array
            A vector of length `params+columns` with the highest values that can take.
        _params : dict
            Dict with the parameter values.
        const : dict
            Dict with the constants values.
        colsnames : list of str
            List with the columns names.
        """
        
        if type(params) is not dict:
            raise Exception("params must be of type dict !!!")

        self._min = np.array([(0 if params[x]["type"] is Population.CATEGORICAL else params[x]["range"][0]) for x in params if params[x]["type"] is not Population.CONSTANT])
        self._max = np.array([(len(params[x]["range"]) if params[x]["type"] is Population.CATEGORICAL else params[x]["range"][1]) for x in params if params[x]["type"] is not Population.CONSTANT])

        self._params = dict((x, params[x]) for x in params if params[x]["type"] is not Population.CONSTANT)
        self.const = dict((x, params[x]["value"]) for x in params if params[x]["type"] is Population.CONSTANT)

        self.colsnames = (columns if type(columns) is list else columns.tolist()) if hasattr(columns, '__iter__') else [f"col_{i}" for i in range(columns)]
        self._min = np.concatenate((self._min, np.zeros(len(self.colsnames))), axis=0)
        self._max = np.concatenate((self._max, np.ones(len(self.colsnames))), axis=0)

        self._pos_n, self._pos_c = list(), list()
        for i, x in enumerate(self._params):
            if self._params[x]["type"] is Population.CATEGORICAL:
                self._pos_c.append(i)
            else:
                self._pos_n.append(i)




        def _trans_mut():

            def compute_feature_probability(threshold):
                p = np.random.uniform(low=0, high=1)  # Número aleatorio que decidirá si tenemos que ponerlo a True o no.
                if p <= threshold: #Si tenemos que tener un True
                    return np.random.uniform(low=0.5, high=1)
                else: #Tenemos que tener un false
                    return np.random.uniform(low=0, high=0.5)

            t = list()
            gen = list()
            for x in self.paramsnames:
                if params[x]["type"] == Population.INTEGER:
                    t.append(np.vectorize(lambda x: int(x), otypes=[int]))
                    gen.append(lambda y, x=x, **kwargs: np.random.randint(low=self._min[y], high=self._max[y]))
                elif params[x]["type"] == Population.FLOAT:
                    t.append(np.vectorize(lambda x: float(x), otypes=[float]))
                    gen.append(lambda y, x=x, **kwargs: np.random.uniform(low=self._min[y], high=self._max[y]))
                elif params[x]["type"] == Population.POWER:
                    t.append(np.vectorize(lambda x: pow(10,x), otypes=[float])) # TODO: No tengo claro que esté bien.
                    gen.append(lambda y, x=x, **kwargs: pow(10,np.random.randint(low=self._min[y], high=self._max[y])))
                elif params[x]["type"] == Population.CATEGORICAL:
                    t.append(np.vectorize(lambda y, x=x: y if type(y) is str else params[x]["range"][int(np.trunc(y))], otypes=[str]))
                    gen.append(lambda y, x=x, **kwargs: np.random.randint(low=self._min[y], high=self._max[y]))
            t.extend([lambda x: x>0.5]*len(self.colsnames))
            #gen.extend([lambda y, x=x, **kwargs: np.random.uniform(low=self._min[y], high=self._max[y]) <= kwargs["feat_mut_thres"]]*len(self.colsnames))
            gen.extend([lambda y, x=x, **kwargs: compute_feature_probability(kwargs["feat_mut_thres"])]*len(self.colsnames))

            # We have to avoid 0-dimensional numpy arrays. Otherwise, some algorithms that perform type
            # checks will fail since, for instance, they receive an integer as a 0-dimensional array, but expect an
            # integer.

            def aux(x):
                if len(x.shape) > 1:
                    return np.array(list(map(lambda f, c: f(x[:, c]), t, range(0, x.shape[1]))), dtype=object).T
                else:
                    return list(
                        map(lambda i: i[1][0](i[1][1]).item() if i[0] < len(self.paramsnames) else i[1][0](i[1][1]),
                            enumerate(zip(t, x))))
            return aux, gen


        self._transformers, self.random_gen = _trans_mut()
        

        if population is not None:
            if type(population) is not np.ndarray or len(population.shape) < 2:
                raise Exception("Popularion is not a numpy matrix")
            self.population = population

    @property
    def population(self):
        return self._transformers(self._pop)

    @population.setter
    def population(self, population):
        self._pop = np.apply_along_axis(lambda x: x.astype(object), 1, population.astype(object))

    @property
    def paramsnames(self):
        return list(self._params.keys())

    def __getitem__(self, key):
        return self._pop[key]

    def __setitem__(self, key, newvalue):
        self._pop[key] = newvalue

    def getChromosome(self, key):
        r"""
        This method returns a chromosome from the population. 

        Parameters
        ----------
        key : int
            Chromosome row index .

        Returns
        -------
        Chromosome
            A `Chromosome` object.
        """
        data = self._transformers(self._pop[key, :])
        return Chromosome(data[:len(self.paramsnames)], self.paramsnames, self.const, data[len(self.paramsnames):], self.colsnames)

    # Method that updates the population to satisfy the feat_thres
    def update_to_feat_thres(self, popSize, feat_thres):
        for i in range(popSize): #For each chromosome
            for j in range(len(self._params),len(self.colsnames) + len(self._params)): # Each feature
                p = np.random.uniform(low=0, high=1) #Random number in interval [0,1]
                if p <= feat_thres and self._pop[i,j] < 0.5: # if p <= self.feat_thres, the feature must be true
                    self._pop[i, j] += 0.5
                elif p > feat_thres and self._pop[i,j] >= 0.5: # if p > self.feat_thres, the feature must be false
                    self._pop[i, j] = self._pop[i, j] - 0.5
    
class Chromosome:

    # @autoassign
    def __init__(self, params, name_params, const, cols, name_cols):
        r"""
        This class defines a chromosome which includes the hyperparameters, the constant values, and the feature selection.


        Parameters
        ----------
        params : numpy.array
            The algorithm hyperparameter values.
        name_params : list of str
            The names of the hyperparameters.
        const : numpy.array
            A dictionary with the constants to include in the chomosome.
        cols : numpy.array
            The probabilities for selecting the input features (selected if prob>0.5).
        name_cols : list of str
            The names of the input features.

        Attributes
        ----------
        params : dict
            A dictionary with the parameter values (hyperparameters and constants).
        columns : numpy.array of bool
            A boolean vector with the selected features.
        """
        self._params = params
        self.name_params = name_params
        self.const = const
        self._cols = cols
        self.name_cols = name_cols

    @property
    def params(self):
        return {**dict(zip(self.name_params, self._params)), **self.const}

    @property
    def columns(self):
        return self._cols
