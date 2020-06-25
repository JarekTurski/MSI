from strlearn.evaluators import TestThenTrain
from strlearn.streams import StreamGenerator
from sklearn.metrics import accuracy_score
from strlearn.ensembles import SEA, AWE, WAE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import math as mat
#import AUE1

#mozna zrobic zeby ladniej wypisywalo te wyniki, w 3 kolumnach


rnd_st = 10 #bedziemy podstawiac 10,20,30...100

#klasyfikatory
clf1 = SEA(base_estimator = GaussianNB())
clf2 = AWE(base_estimator = GaussianNB())
clf3 = WAE(base_estimator = GaussianNB())
#clf4 = AUE1(ClassifierMixin, BaseEnsemble)
clfs = (clf1, clf2, clf3)

#ewaluator z odpowiednia metryka
evaluator = TestThenTrain(metrics=accuracy_score) 

#przygotowanie pliku z wynikami
f_sudden = open("wynikidryfnagly.csv", "a")
f_gradual = open("wynikidryfgradualny.csv", "a")
f_incremental = open("wynikidryfinkrementalny", "a")

#usrednianie po n_chunks i wypisywanie wynikow, osobne pliki dla kazdego dryfu
#wyniki sa zapisywane w jednej kolumnie -> 
#ciag: srednia_str1_clf1, srednia_str1_clf2, srednia_str1_clf3, srednia_str2_clf1, ...
for rnd_st in range(10,110,10): #rnd_st przechowuje aktualna wartosc random_state
    #wypisywanie wynikow dla dryfu naglego dla wszystkich klasyfikatorow
    str_sudden =  StreamGenerator(n_drifts=1, random_state=rnd_st)
    evaluator.process(str_sudden, clfs)
    array2d = evaluator.scores.reshape(249,3)
    resultsmean = np.mean(array2d, axis=0)
    np.savetxt(f_sudden, resultsmean, delimiter=",", fmt='%0.3f')

    #wypisywanie wynikow dla dryfu gradualnego dla wszystkich klasyfikatorow
    str_gradual = StreamGenerator(n_drifts=1, concept_sigmoid_spacing=5, random_state=rnd_st)
    evaluator.process(str_gradual, clfs)
    array2d = evaluator.scores.reshape(249,3)
    resultsmean = np.mean(array2d, axis=0)
    np.savetxt(f_gradual, resultsmean, delimiter=",", fmt='%0.3f')

    #wypisywanie wynikow dla dryfu inkremetalnego dla wszystkich klasyfikatorow
    str_incremental = StreamGenerator(n_drifts=1, concept_sigmoid_spacing=5, incremental=True, random_state=rnd_st)
    evaluator.process(str_incremental, clfs)
    array2d = evaluator.scores.reshape(249,3)
    resultsmean = np.mean(array2d, axis=0)
    np.savetxt(f_incremental, resultsmean, delimiter=",", fmt='%0.3f')

f_sudden.close()
f_gradual.close()
f_incremental.close()