# -*- coding: utf-8 -*-

import numpy  as np


# new function for monitoring
def parsimony_monitor(iter, current_best_score, current_best_complexity, fitnessval, bestfitnessVal, 
                      bestcomplexity, minutes_gen, digits=6, *args):
    r"""Functions for monitoring HYB-PARSIMONY algorithm evolution

    Functions to print summary statistics of fitness values at each iteration of a GA search.

    Parameters
    ----------
    iter: int
        Iteration.
    current_best_score: float
        The best score in the whole process (score of the best model).
    current_best_complexity: float
        The complexity of the best model in the whole process.
    fitnessval: list
        Fitness values of the population in that iteration.
    bestfitnessVal: float
        Best fitness value in this iteration (score of the best model in that iteration)
    bestcomplexity: float
        The complexity of the best model in that iteration.
    minutes_gen: float
        Time in minutes of that iteration.
    digits : int
        Minimal number of significant digits.
    *args :
        Further arguments passed to or from other methods.
    """

    fitnessval = fitnessval[~np.isnan(fitnessval)]

    print(" ".join([f"Best model -> Score = {round(current_best_score, digits)}".center(16 + digits),
                     f"Complexity = {round(current_best_complexity, 2):,}".center(12),
                    f"\nIter = {iter} -> MeanVal = {round(np.mean(fitnessval), digits)}".center(16 + digits),
                    f"ValBest = {round(bestfitnessVal, digits)}".center(16 + digits),
                    # f"TstBest = {round(bestfitnessTst, digits)}".center(16 + digits),
                    f"ComplexBest = {round(bestcomplexity, 2):,}".center(12),
                    f"Time(min) = {round(minutes_gen, digits)}".center(7)]) + "\n")
 


# Duda si es todo el rato con x1
# Equivalencia a fivenum es np.percentile(aux, [0, 25, 50, 75, 100])

def parsimony_summary(fitnessval, complexity, *args):
    x1 = fitnessval[~np.isnan(fitnessval)]
    q1 = np.percentile(x1, [0, 25, 50, 75, 100])
    # x2 = fitnesstst[~np.isnan(fitnesstst)]
    # q2 = np.percentile(x1, [0, 25, 50, 75, 100])
    x3 = complexity[~np.isnan(complexity)]
    q3 = np.percentile(x1, [0, 25, 50, 75, 100])

    return q1[4], np.mean(x1), q1[3], q1[2], q1[1], q1[0], q3[
        4], np.mean(x3), q3[3], q3[2], q3[1], q3[0]


