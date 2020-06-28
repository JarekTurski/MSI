from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaseEnsemble
import sklearn.utils.validation as suvalid #training utilities module
from sklearn.model_selection import cross_val_score
import numpy as np

# metody partial_fit oraz predict sa uzywane w TestThenTrain.process

class AUE(ClassifierMixin, BaseEnsemble):

    #konstruktor
    def __init__(self, base_estimator, epsilon = 0.00000001, k = 10, cv=5):
        self.base_estimator = base_estimator
        self.epsilon = epsilon
        self.k = k
        self.cv= cv

    def fit(self, X, y):
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes):

        X, y = suvalid.check_X_y(X, y)  #sprawdzanie poprawnosci danych wejsciowych
       
        self.X_, self.y_ = X, y
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        #obliczenie prawdopodobienstwa
        y_unique = np.unique(y)
        p_c = np.array([])
        for i in range(0, len(y_unique)):
            p_c = np.append(p_c, np.count_nonzero(y == y_unique[i]))

        probability = (p_c/ len(y))

        #komitet klasyfikatorow
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []

        #obliczanie mser
        mser = np.sum(probability * np.power((1 - probability), 2))

        C_prime = clone(self.base_estimator).fit(self.X_, self.y_) #uczenie Ci na X 

        #obliczyc MSE dla Ci, walidacja krzyzowa na X
        cv_score = np.mean(-cross_val_score(C_prime, self.X_, self.y_, cv=self.cv, scoring='neg_mean_squared_error')) 

        w_prime = 1 / (cv_score + self.epsilon)  #obliczenie wagi

        #petla do obliczania wag pozostalych klasyfikatorow z komitetu
        C_scores = [(1 / (mean_squared_error(Ci.predict(self.X_), self.y_) + self.epsilon), Ci, False) for Ci in self.ensemble_]
        C_scores.append((w_prime, C_prime, True))

        #k klasyfikatorow o najwyzszej wadze - sortowanie 
        C_scores.sort(key=lambda element: element[0], reverse=True)
        C_scores = C_scores[:self.k]
            
        #petla: update k klasyfikatorow o najwiekszej wadze
        for classifier in C_scores:
            if classifier[0] > 1 / mser and classifier[2] == False:
                classifier[1].partial_fit(self.X_, self.y_)

        self.ensemble_ = [c[1] for c in C_scores]

        return self

    def predict(self, X):
        suvalid.check_is_fitted(self, "classes_") #sprawdzenie poprawnosci estymatora >> check_is_fitted(self)
        X = suvalid.check_array(X)  #sprawdzenie poprawnosci tablicy i zwrocenie poprawnej
        
        # GLOSOWANIE MIEKKIE
        esm = np.mean(np.array([clf.predict_proba(X) for clf in self.ensemble_]), axis=0)
        average = np.mean(esm, axis=0)  #wyliczenie sredniej wartosci wsparcia
        prediction_array = np.argmax(esm, axis=1)  #wskazanie etykiet

        return self.classes_[prediction_array] #zwrocenie predykcji
        
    