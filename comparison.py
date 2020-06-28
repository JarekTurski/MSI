from strlearn.evaluators import TestThenTrain
from strlearn.streams import StreamGenerator
from sklearn.metrics import accuracy_score
from strlearn.ensembles import SEA, AWE, WAE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import math as mat
from scipy.stats import ttest_ind
from tabulate import tabulate
#import AUE1

#petle!

rnd_st = 10 #bedziemy podstawiac 10,20,30...100

#Classificators
clf1 = SEA(base_estimator = GaussianNB())
clf2 = AWE(base_estimator = GaussianNB())
clf3 = WAE(base_estimator = GaussianNB())
#clf4 = AUE1(ClassifierMixin, BaseEnsemble)
clfs = (clf1, clf2, clf3) #dopisac AUE

#Evaluator with accuracy_score metrics
evaluator = TestThenTrain(metrics=accuracy_score)

#Arrays for each drift(columns: clf, rows: streams)
scores_sudden = np.zeros(shape=(10,len(clfs)))  #zamienic na empty
scores_gradual = np.zeros(shape=(10,len(clfs)))
scores_incremental = np.zeros(shape=(10,len(clfs)))

#files for scores and analysys (different file for each drift)
open("wynikidryfnagly.csv","w").close()
open("wynikidryfgradualny.csv","w").close()
open("wynikidryfinkrementalny.csv","w").close()
f_sudden = open("wynikidryfnagly.csv", "a")
f_gradual = open("wynikidryfgradualny.csv", "a")
f_incremental = open("wynikidryfinkrementalny.csv", "a")

#Scores mean for each stream
for rnd_st in range(10,110,10): #rnd_st przechowuje aktualna wartosc random_state
    #sudden drift
    str_sudden =  StreamGenerator(n_drifts=1, random_state=rnd_st)
    evaluator.process(str_sudden, clfs)
    array2d = evaluator.scores.reshape(249, len(clfs))
    scores_sudden[int(rnd_st/10 - 1)] = np.mean(array2d, axis=0)

    #gradual drift
    str_gradual = StreamGenerator(n_drifts=1, concept_sigmoid_spacing=5, random_state=rnd_st)
    evaluator.process(str_gradual, clfs)
    array2d = evaluator.scores.reshape(249,len(clfs))
    scores_gradual[int(rnd_st/10 - 1)] = np.mean(array2d, axis=0)

    #incremental drift
    str_incremental = StreamGenerator(n_drifts=1, concept_sigmoid_spacing=5, incremental=True, random_state=rnd_st)
    evaluator.process(str_incremental, clfs)
    array2d = evaluator.scores.reshape(249,len(clfs))
    scores_incremental[int(rnd_st/10 - 1)] = np.mean(array2d, axis=0)

#write mean scores to file
headers = ["SEA", "AWE", "WAE"] #dopisac AUE
s_scores_table = tabulate(scores_sudden, headers, floatfmt=".2f")
f_sudden.write("Mean score for each of 10 streams and 4 classificators:\n")
f_sudden.write(s_scores_table) 

g_scores_table = tabulate(scores_gradual, headers, floatfmt=".2f")
f_gradual.write("Mean score for each of 10 streams and 4 classificators:\n")
f_gradual.write(g_scores_table)

i_scores_table = tabulate(scores_incremental, headers, floatfmt=".2f")
f_incremental.write("Mean score for each of 10 streams and 4 classificators:\n")
f_incremental.write(i_scores_table)

#t-statistics and p-value
alfa = .05
t_statistic_sudden = np.zeros((len(clfs), len(clfs)))
p_value_sudden = np.zeros((len(clfs), len(clfs)))

t_statistic_gradual = np.zeros((len(clfs), len(clfs)))
p_value_gradual = np.zeros((len(clfs), len(clfs)))

t_statistic_incremental = np.zeros((len(clfs), len(clfs)))
p_value_incremental = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic_sudden[i, j], p_value_sudden[i, j] = ttest_ind(scores_sudden[i], scores_sudden[j])
        t_statistic_gradual[i, j], p_value_gradual[i, j] = ttest_ind(scores_gradual[i], scores_gradual[j])
        t_statistic_incremental[i, j], p_value_incremental[i, j] = ttest_ind(scores_incremental[i], scores_incremental[j])

headers = ["SEA", "AWE", "WAE"] #dopisac AUE
names_column = np.array([["SEA"], ["AWE"], ["WAE"]]) #dopisac AUE
t_statistic_table_sudden = np.concatenate((names_column, t_statistic_sudden), axis=1)
t_statistic_table_sudden = tabulate(t_statistic_table_sudden, headers, floatfmt=".2f")
p_value_table_sudden = np.concatenate((names_column, p_value_sudden), axis=1)
p_value_table_sudden = tabulate(p_value_table_sudden, headers, floatfmt=".2f")
f_sudden.write("\n t-statistic:\n")
f_sudden.write(t_statistic_table_sudden)
f_sudden.write("\n p-value:\n")
f_sudden.write(p_value_table_sudden)

t_statistic_table_gradual = np.concatenate((names_column, t_statistic_gradual), axis=1)
t_statistic_table_gradual = tabulate(t_statistic_table_gradual, headers, floatfmt=".2f")
p_value_table_gradual = np.concatenate((names_column, p_value_gradual), axis=1)
p_value_table_gradual = tabulate(p_value_table_gradual, headers, floatfmt=".2f")
f_gradual.write("\n t-statistic:\n")
f_gradual.write(t_statistic_table_gradual)
f_gradual.write("\n p-value:\n")
f_gradual.write(p_value_table_gradual)

t_statistic_table_incremental = np.concatenate((names_column, t_statistic_incremental), axis=1)
t_statistic_table_incremental = tabulate(t_statistic_table_incremental, headers, floatfmt=".2f")
p_value_table_incremental = np.concatenate((names_column, p_value_incremental), axis=1)
p_value_table_incremental = tabulate(p_value_table_incremental, headers, floatfmt=".2f")
f_incremental.write("\n t-statistic:\n")
f_incremental.write(t_statistic_table_incremental)
f_incremental.write("\n p-value:\n")
f_incremental.write(p_value_table_incremental)

#advantage
advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic_sudden > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
f_sudden.write("\n Advantage:\n")
f_sudden.write(advantage_table)
advantage_g = np.zeros((len(clfs), len(clfs)))
advantage_g[t_statistic_gradual > 0] = 1
advantage_table_g = tabulate(np.concatenate(
    (names_column, advantage_g), axis=1), headers)
f_gradual.write("\n Advantage:\n")
f_gradual.write(advantage_table_g)
advantage_i = np.zeros((len(clfs), len(clfs)))
advantage_i[t_statistic_incremental > 0] = 1
advantage_table_i = tabulate(np.concatenate(
    (names_column, advantage_i), axis=1), headers)
f_incremental.write("\n Advantage:\n")
f_incremental.write(advantage_table_i)

#statistical significance
significance = np.zeros((len(clfs), len(clfs)))
significance[p_value_sudden <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
f_sudden.write("\n Statistical significance (alpha=0,05):\n")
f_sudden.write(significance_table)
significance_g = np.zeros((len(clfs), len(clfs)))
significance_g[p_value_gradual <= alfa] = 1
significance_table_g = tabulate(np.concatenate(
    (names_column, significance_g), axis=1), headers)
f_gradual.write("\n Statistical significance (alpha=0,05):\n")
f_gradual.write(significance_table_g)
significance_i = np.zeros((len(clfs), len(clfs)))
significance_i[p_value_incremental <= alfa] = 1
significance_table_i = tabulate(np.concatenate(
    (names_column, significance_i), axis=1), headers)
f_incremental.write("\n Statistical significance (alpha=0,05):\n")
f_incremental.write(significance_table_i)

#final result
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
f_sudden.write("\n Statistically significantly better:\n")
f_sudden.write(stat_better_table)
stat_better_g = significance_g * advantage_g
stat_better_table_g = tabulate(np.concatenate(
    (names_column, stat_better_g), axis=1), headers)
f_gradual.write("\n Statistically significantly better:\n")
f_gradual.write(stat_better_table_g)
stat_better_i = significance_i * advantage_i
stat_better_table_i = tabulate(np.concatenate(
    (names_column, stat_better_i), axis=1), headers)
f_incremental.write("\n Statistically significantly better:\n")
f_incremental.write(stat_better_table_i)

f_sudden.close()
f_gradual.close()
f_incremental.close()
