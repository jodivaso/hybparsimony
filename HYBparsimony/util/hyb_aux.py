import numpy as np

from HYBparsimony import order
from HYBparsimony.lhs import geneticLHS, improvedLHS, maximinLHS, optimumLHS, randomLHS

def _population(pop, seed_ini, popSize, type_ini_pop="randomLHS", ):
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

    nvars = len(pop._params) + len(pop.colsnames)
    if type_ini_pop == "randomLHS":
        population = randomLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "geneticLHS":
        population = geneticLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "improvedLHS":
        population = improvedLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "maximinLHS":
        population = maximinLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "optimumLHS":
        population = optimumLHS(popSize, nvars, seed=seed_ini)
    elif type_ini_pop == "random":
        population = (np.random.rand(popSize * nvars) * (nvars - popSize) + popSize).reshape(
            popSize * nvars, 1)

    # Scale matrix with the parameters range
    population = population * (pop._max - pop._min)
    population = population + pop._min

    # If one wants to convert features to binary:
    #population[:, len(pop._params):nvars] = population[:,len(pop._params):nvars] <= feat_thres

    return population


def _rerank(fitnessval, complexity, popSize, rerank_error, preserve_best=True):
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

    cost1 = fitnessval.copy().astype(float)
    cost1[np.isnan(cost1)] = np.NINF

    sort = order(cost1, decreasing=True)
    cost1 = cost1[sort]
    complexity = complexity.copy()
    complexity[np.isnan(complexity)] = np.Inf
    complexity = complexity[sort]
    position = sort

    # start
    if preserve_best and len(cost1)>1:
        pos1 = 1
        pos2 = 2
        error_posic = cost1[1]
    else:
        pos1 = 0
        pos2 = 1
        error_posic = cost1[0]
    cambio = False

    while not pos1 == popSize:
        # Obtaining errors
        if pos2 >= popSize:
            if cambio:
                pos2 = pos1 + 1
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
        if error_dif < rerank_error:

            # If there is not difference between errors swap if Size2nd < SizeFirst
            size_indiv1 = complexity[pos1]
            size_indiv2 = complexity[pos2]
            if size_indiv2 < size_indiv1:
                cambio = True

                swap_indiv = cost1[pos1]
                cost1[pos1] = cost1[pos2]
                cost1[pos2] = swap_indiv

                complexity[pos1], complexity[pos2] = complexity[pos2], complexity[pos1]

                position[pos1], position[pos2] = position[pos2], position[pos1]

                # if self.verbose == 2:
                #     print(f"SWAP!!: pos1={pos1}({size_indiv1}), pos2={pos2}({size_indiv2}), error_dif={error_dif}")
                #     print("-----------------------------------------------------")
            pos2 = pos2 + 1

        elif cambio:
            cambio = False
            pos2 = pos1 + 1
        else:
            pos1 = pos1 + 1
            pos2 = pos1 + 1
            error_dif2 = abs(cost1[pos1] - error_posic)
            if not np.isfinite(error_dif2):
                error_dif2 = np.Inf
            if error_dif2 >= rerank_error:
                error_posic = cost1[pos1]
    return position


def _crossover(population, velocities, fitnessval, complexity, parents_indexes, children_indexes, alpha=0.1, perc_to_swap=0.5):

    #p = parents.copy()
    c = children_indexes.copy()

    number_children = len(children_indexes) # Should be 1 or 2.

    velocities_parents = velocities[parents_indexes].copy()
    velocities_children = velocities[parents_indexes].copy()
    velocities_children = velocities_children[0:number_children] # This makes velocities_children to have one row if there is only 1 child.

    parents = population._pop[parents_indexes]

    children = population._pop[parents_indexes].copy()  # Children will be the crossover of the parents
    children = children[0:number_children] # This makes children to have one row if there is only 1 child.

    pos_param_n = population._pos_n
    pos_param_c = population._pos_c
    pos_features = np.array(
        list(range(len(population._params), len(population._params) + len(population.colsnames))))

    # Heuristic Blending for parameters not categoricals
    Betas = np.random.uniform(size=len(pos_param_n), low=0, high=1) * (1 + 2 * alpha) - alpha  # 1+alpha*2??????
    children[0, pos_param_n] = parents[0, pos_param_n] - Betas * parents[0, pos_param_n] + Betas * parents[
        1, pos_param_n]  ## MAP??
    velocities_children[0, pos_param_n] = velocities_parents[0, pos_param_n] - Betas * velocities_parents[0, pos_param_n] + Betas * velocities_parents[
        1, pos_param_n]
    if number_children > 1:
        children[1, pos_param_n] = parents[1, pos_param_n] - Betas * parents[1, pos_param_n] + Betas * parents[
            0, pos_param_n]
        velocities_children[1, pos_param_n] = velocities_parents[1, pos_param_n] - Betas * velocities_parents[1, pos_param_n] + Betas * velocities_parents[
            0, pos_param_n]


    # Random swapping for categorical parameters
    swap_param_c = np.random.uniform(size=len(pos_param_c), low=0, high=1) >= perc_to_swap
    if np.sum(swap_param_c) > 0:
        parameters_c_parent1 = parents[0, pos_param_c]
        parameters_c_parent2 = parents[1, pos_param_c]
        velocities_c_parent1 = velocities_parents[0, pos_param_c]
        velocities_c_parent2 = velocities_parents[1, pos_param_c]
        pos_param_c = np.array(pos_param_c)[swap_param_c]
        children[0, pos_param_c] = parameters_c_parent2[swap_param_c]
        velocities_children[0, pos_param_c] = velocities_c_parent2[swap_param_c]
        if number_children > 1:
            children[1, pos_param_c] = parameters_c_parent1[swap_param_c]
            velocities_children[1, pos_param_c] = velocities_c_parent1[swap_param_c]

    # Random swapping for features
    swap_param = np.random.uniform(size=len(population.colsnames), low=0, high=1) >= perc_to_swap
    if np.sum(swap_param) > 0:
        features_parent1 = parents[0, pos_features]
        features_parent2 = parents[1, pos_features]
        velocities_parent1 = velocities_parents[0, pos_features]
        velocities_parent2 = velocities_parents[1, pos_features]
        pos_features = pos_features[swap_param]
        children[0, pos_features] = features_parent2[swap_param]
        velocities_children[0, pos_features] = velocities_parent2[swap_param]
        if number_children > 1:
            children[1, pos_features] = features_parent1[swap_param]
            velocities_children[1, pos_features] = velocities_parent1[swap_param]

    # correct params that are outside (min and max)
    thereis_min = children[0] < population._min
    children[0, thereis_min] = population._min[thereis_min]
    if number_children > 1:
        thereis_min = children[1] < population._min
        children[1, thereis_min] = population._min[thereis_min]

    thereis_max = children[0] > population._max
    children[0, thereis_max] = population._max[thereis_max]
    if number_children > 1:
        thereis_max = (children[1] > population._max)
        children[1, thereis_max] = population._max[thereis_max]

    if number_children > 1:
        aux = np.empty(2)
    else:
        aux = np.empty(1)

    aux[:] = np.nan

    population[c] = children
    fitnessval[c] = aux.copy()
    #fitnesstst[c] = aux.copy()
    complexity[c] = aux.copy()

    velocities[c] = velocities_children