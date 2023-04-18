import numpy as np
from sklearn.linear_model import Lasso, RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, mean_absolute_error
from sklearn.datasets import load_diabetes, load_iris
from sklearn.neighbors import KNeighborsRegressor
from HYBparsimony import Population, HYBparsimony, order
from HYBparsimony.hybparsimony import default_cv_score_classification
from HYBparsimony.util import knn_complexity
from HYBparsimony.util.models import Logistic_Model

if __name__ == "__main__":

    # REGRESIÓN
    # Cargo un dataset de regresión
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    ###############################################################
    #                       EJEMPLO BÁSICO                        #
    ###############################################################
    # HYBparsimony_model = HYBparsimony()
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                       EJEMPLO OTRO SCORING                  #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(scoring="neg_mean_absolute_error")
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("MAE test", mean_absolute_error(y_test, preds))

    ###############################################################
    #                          EJEMPLO OTRO CV                    #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(cv=RepeatedKFold(n_splits=10, n_repeats=5))
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                EJEMPLO OTRO SCORING Y OTRO CV               #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(scoring="neg_mean_absolute_error", cv=RepeatedKFold(n_splits=10, n_repeats=5))
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("MAE test", mean_absolute_error(y_test, preds))


    ###############################################################
    #                     EJEMPLO CUSTOM_EVAL                     #
    ###############################################################

    # def custom_fun(estimator, X, y): # CV con repeatedKfold.
    #     return cross_val_score(estimator, X, y, cv=RepeatedKFold(n_splits=10, n_repeats=5))
    #
    # # Este con paralelismo NO funciona (sí funciona si el custom_fun lo definimos fuera del if main)
    # HYBparsimony_model = HYBparsimony(n_jobs=1, custom_eval_fun=custom_fun)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                     EJEMPLO OTRO ALGORITMO                  #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(algorithm="MLPRegressor")
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))

    ###############################################################
    #                     EJEMPLO FEATURES PARAMS                 #
    ###############################################################
    # HYBparsimony_model = HYBparsimony(features = diabetes.feature_names)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict(X_test)
    # print("RMSE test", mean_squared_error(y_test, preds))


    ###############################################################
    #                     EJEMPLO NUEVO ALGORITMO                 #
    ###############################################################

    # KNN_Model = {"estimator": KNeighborsRegressor,
    #              "complexity": knn_complexity,
    #              "n_neighbors": {"range": (1, 10), "type": Population.INTEGER}
    #              }
    # HYBparsimony_model = HYBparsimony(algorithm=KNN_Model)
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # print(HYBparsimony_model.best_model)
    # print(HYBparsimony_model.selected_features)


    # CLASIFICACIÓN

    # Cargo un dataset de clasificación
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=12345)

    ###############################################################
    #                       EJEMPLO BÁSICO                        #
    ###############################################################

    # HYBparsimony_model = HYBparsimony()
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.2)
    # preds = HYBparsimony_model.predict_proba(X_test)
    # print("Log loss test", log_loss(y_test, preds))

    ###############################################################
    #                     EJEMPLO CUSTOM_EVAL                     #
    ###############################################################

    # def custom_fun(estimator, X, y):
    #     return cross_val_score(estimator, X, y, scoring="accuracy")
    #
    # HYBparsimony_model = HYBparsimony(n_jobs=1,custom_eval_fun=custom_fun) # Este con paralelismo NO funciona
    # HYBparsimony_model.fit(X_train, y_train, time_limit=0.5)
    # preds = HYBparsimony_model.predict(X_test)
    # print("Accuracy test", accuracy_score(y_test, preds))


    ###############################################################
    #                        EJEMPLO OTRO CV
    ###############################################################
    for i in range(1000):
        print("Attempt", i)
        HYBparsimony_model = HYBparsimony(seed_ini=i, cv=RepeatedKFold(n_splits=10, n_repeats=5))
        HYBparsimony_model.fit(X_train, y_train, time_limit=1)
        preds = HYBparsimony_model.predict(X_test)
        print("Accuracy test", accuracy_score(y_test, preds))

# fitnessval = np.array([        np.nan,         np.nan,         np.nan,         np.nan,         np.nan, -0.08044018,
#  -0.10418209, -0.08177489,         np.nan,         np.nan, -0.08123836,         np.nan,
#          np.nan,         np.nan, -0.08634382,         np.nan,         np.nan,         np.nan,
#          np.nan,         np.nan,         np.nan,         np.nan,         np.nan, -0.09825784,
#          np.nan,         np.nan, -0.10878671 ,        np.nan, -0.11083702, -0.08347538,
#          np.nan,         np.nan, -0.0849746,  -0.08306158, -0.1345692,  -0.07867179,
#          np.nan,         np.nan,         np.nan, -0.10950051])
#
# sort = order(fitnessval, kind='heapsort', decreasing=True, na_last=True)
# print(sort)
# print(fitnessval[sort])
# print(fitnessval[35])
# print(np.argsort(fitnessval))
# print(fitnessval[np.argsort(fitnessval)])

# population_sin_sort=np.array(
# [[6.762656407373061,0.8269611543324937,0.9641179178387247,1.0
# ,0.9257152284514165],
# [5.41155998327111,0.9796483613645612,0.08238534208129322
# ,0.5949595368892593,0.5521711091954123],
# [7.58939743670119,1.0,0.0,0.4067871464274668,0.43681445640419814],
# [5.440492744150532,0.9796483613645612,0.27191320763885996
# ,0.5949595368892593,0.515150668124545],
# [7.6146991574564495,1.0,0.25602688676433727,0.55552338679523
# ,0.32188295522302146],
# [5.51824007712098,0.9796483613645612,0.27191320763885996
# ,0.5949595368892593,0.515150668124545],
# [7.923670517402218,0.7874358354716552,0.6310972267962642
# ,0.6196168710215366,0.2352816806424755],
# [4.911518431922703,0.45655417433932466,0.2334117188907634
# ,0.29016139738121344,0.49591053047227174],
# [5.116629605423196,0.9796483613645612,0.27191320763885996
# ,0.5949595368892593,0.7690440956283475],
# [3.780108169045673,1.0,0.6110436950325407,1.0,0.7826390935801928],
# [7.203433413525925,0.9608434866259609,0.08238534208129322
# ,0.524221307517362,0.8701567501875269],
# [4.5676297219340665,1.0,0.17137465551218944,0.8323084592077985,1.0],
# [7.550833340899695,0.6659216597171609,0.49747358079379145
# ,0.5450435423414145,0.8509265153505668],
# [6.008290667789926,0.9608434866259609,0.7880127500723779
# ,0.6713769333296706,0.8701567501875269],
# [6.010947442851515,0.9796483613645612,0.21090053253250557
# ,0.5949595368892593,0.8204661715286402],
# [6.723457895170599,1.0,1.0,0.4939751929633813,0.991808192699919],
# [6.551301078345082,0.6524463444888998,0.25825848296756165
# ,0.6749307339543309,0.8284401642851095],
# [6.322306980200229,0.9796483613645612,0.032909725410675494
# ,0.5949595368892593,0.8204661715286402],
# [5.8571159628861365,0.9796483613645612,0.8053422304127438
# ,0.5949595368892593,0.8204661715286402],
# [5.355066178554452,0.9608434866259609,0.08238534208129322
# ,0.5949595368892593,0.8204661715286402],
# [5.643718867158085,0.5205093738173158,0.08238534208129322
# ,0.7736948301164821,0.515150668124545],
# [5.8552163464024805,1.0,0.830202864980933,0.6190149343433342
# ,0.8107194780875142],
# [5.923654532873563,0.7491840250117604,0.806455325977966
# ,0.33627337872856966,0.6238206844090768],
# [6.662701482436237,0.8003096663037317,0.07753720464579139
# ,0.38883492698224553,0.6699945267394906],
# [5.729709556852703,0.9796483613645612,0.27191320763885996
# ,0.5949595368892593,0.8204661715286402],
# [5.385400910889984,0.5205093738173158,0.08238534208129322
# ,0.7736948301164821,0.5521711091954123],
# [7.432971951259634,0.1721570989204362,0.16052662671577578
# ,0.6713769333296706,0.8701567501875269],
# [9.712358081027837,0.7171994892022299,0.43389545451505535
# ,0.7431600243233479,0.8284884179675513],
# [7.930592514649812,0.9160598078244344,0.6578746012940068
# ,0.6160550830775705,1.0],
# [6.917121073829755,3.9776421139319234,0.7880127500723779
# ,0.6713769333296706,0.8204661715286402],
# [4.624281798709051,0.44379594203590456,0.9055615612363214
# ,0.45957918682913795,0.6748121382577693],
# [10.0,0.9179183210276138,1.0,0.6467712809175533,0.7557963706272091],
# [6.15361176336235,1.0,0.22774837634783301,0.6035401736246171
# ,0.6671552070225142],
# [7.441553782105536,0.9608434866259609,0.7880127500723779
# ,0.6713769333296706,0.8701567501875269],
# [5.357012013993684,0.8709828213206853,0.24086166031456407
# ,0.6713769333296706,0.5521711091954123],
# [7.3387116968177795,0.9608434866259609,0.7880127500723779
# ,0.6713769333296706,0.8701567501875269],
# [9.55238903549064,0.8518678460694651,0.14760093206916858
# ,0.33222239233954204,0.5404009640874363],
# [7.0514099037629485,0.5357141170957668,0.788076353018923
# ,0.4210834522936349,0.8359237327437448],
# [8.327224289398993,0.8001775694609262,0.6152725795497765
# ,0.640201847087874,0.6582603600351566],
# [5.7188004239785215,0.5205093738173158,0.27191320763885996
# ,0.5949595368892593,0.515150668124545]])
#
# models_sin_sort = np.array([LogisticRegression(C=6.762656407373061),
# LogisticRegression(C=5.41155998327111),
# LogisticRegression(C=7.58939743670119),
# LogisticRegression(C=5.440492744150532),
# LogisticRegression(C=7.6146991574564495),
# LogisticRegression(C=5.51824007712098),
# LogisticRegression(C=7.923670517402218),
# LogisticRegression(C=5.116629605423196),
# LogisticRegression(C=3.780108169045673),
# LogisticRegression(C=7.203433413525925),
# LogisticRegression(C=4.5676297219340665),
# LogisticRegression(C=7.550833340899695),
# LogisticRegression(C=6.008290667789926),
# LogisticRegression(C=6.010947442851515),
# LogisticRegression(C=6.723457895170599),
# LogisticRegression(C=6.551301078345082),
# LogisticRegression(C=6.322306980200229),
# LogisticRegression(C=5.8571159628861365),
# LogisticRegression(C=5.355066178554452),
# LogisticRegression(C=5.643718867158085),
# LogisticRegression(C=5.8552163464024805),
# LogisticRegression(C=5.923654532873563),
# LogisticRegression(C=6.662701482436237),
# LogisticRegression(C=5.729709556852703),
# LogisticRegression(C=5.385400910889984),
# LogisticRegression(C=7.432971951259634),
# LogisticRegression(C=9.712358081027837),
# LogisticRegression(C=7.930592514649812),
# LogisticRegression(C=6.917121073829755),
# LogisticRegression(C=4.624281798709051),LogisticRegression(C=10.0),
# LogisticRegression(C=6.15361176336235),
# LogisticRegression(C=7.441553782105536),
# LogisticRegression(C=5.357012013993684),
# LogisticRegression(C=7.3387116968177795),
# LogisticRegression(C=9.55238903549064),
# LogisticRegression(C=7.0514099037629485),
# LogisticRegression(C=8.327224289398993),
# LogisticRegression(C=5.7188004239785215),
# LogisticRegression(C=4.472153873849367)])
#
# fitnessval = np.array([        np.nan, -0.08722549,         np.nan, -0.08683541 ,        np.nan, -0.08524605,
#  -0.11331457 ,        np.nan,         np.nan,         np.nan, -0.09143413, -0.0789949,
#  -0.08634393, -0.08170286, -0.13972469, -0.08089602, -0.0824617 , -0.08435289,
#          np.nan,         np.nan,         np.nan,         np.nan,         np.nan,         np.nan,
#  -0.08555089,         np.nan, -0.07604595, -0.07854805, -0.0822313 , -0.15946072,
#          np.nan, -0.08449445, -0.08310134  ,       np.nan,         np.nan,         np.nan,
#  -0.13612438 ,        np.nan,         np.nan,         np.nan])
#
# sort = order(fitnessval, kind='heapsort', decreasing=True, na_last=True)
# print(sort)
# print()
#
# for i in range(40):
#     if models_sin_sort[i].C != population_sin_sort[i,0]:
#         print(i, models_sin_sort[i].C, population_sin_sort[i,0])
# sort [26 27 11 15 13 28 16 32 17 31  5 24 12  3  1 10  6 36 14 29  0 39 38 37
#   7 33 34  8 35 18  4  9 19 20 21 22  2 23 25 30]







