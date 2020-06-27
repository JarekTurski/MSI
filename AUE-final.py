from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_val_score
import sklearn.utils.validation as suvalid #training utilities module
import numpy as np


class AUEFinal(ClassifierMixin, BaseEnsemble):

    #konstruktor
    def __init__(self, base_estimator, epsilon = 0.00000001, cv=5, k = 20):
        self.base_estimator = base_estimator
        self.epsilon = epsilon
        self.k = k
        self.cv= cv

    def count_p_c(self,y):
        y_unique = np.unique(y)
        p_c = np.array([])
        for i in range(0, len(y_unique)):
            p_c = np.append(p_c, np.count_nonzero(y == y_unique[i]))
    
        return(p_c/ len(y))

    def partial_fit(self, X, y, classes):

        X, y = suvalid.check_X_y(X, y)  #sprawdzanie poprawnosci danych wejsciowych
       
        self.X_, self.y_ = X, y
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        p_c = count_p_c(y)   #obliczenie prawdopodobienstwa

        if not hasattr(self, "ensemble_"):  #tworzenie tablicy ensemble_ dla k klasyfikatorow o najwiekszej wadze
            self.ensemble_ = []

        mser = np.sum(p_c * np.power((1 - p_c), 2))  #obliczanie MSEr
 
        c_prime = clone(self.base_estimator).fit(self.X_, self.y_)  #uczenie Ci na X 

        #obliczyc MSE dla Ci, walidacja krzyzowa na X
        cv_score = np.mean(cross_val_score(c_prime, self.X_, self.y_, cv=self.cv, scoring='neg_mean_squared_error')) 
        
        w_prime = 1 / (cv_score + self.epsilon)  #obliczenie wagi

        #petla do obliczania wag pozostalych klasyfikatorow z komitetu
        C_scores = [(1 / (mean_squared_error(Ci.predict(self.X_), self.y_) + self.epsilon), Ci, False) for Ci in self.ensemble_]
        C_scores.append((w_prime, c_prime, True))


        #petla: update k klasyfikatorow o najwiekszej wadze

        """
        e = len(self.ensemble_)

        for e in self.ensemble_:
            if self.w_i[e] > (1/mser):
                e.fit(self.X_, self.y_)

        return self
        """



    def predict(self, X):
        suvalid.check_is_fitted(self, "classes_") #sprawdzenie poprawnosci estymatora >> check_is_fitted(self)
        X = suvalid.check_array(X)  #sprawdzenie poprawnosci tablicy i zwrocenie poprawnej
        if X.shape[1] != self.n_features:  #kontrola zgodnosci liczby cechc z danymi uzytymi do uczenia
            raise ValueError("Ilosc cech sie nie zgadza!")
        
        # GLOSOWANIE MIEKKIE
        esm = self.ensemble_support_matrix(X)  #podejmowanie decyzji na podstawie wektorow wsparcia
        average = np.mean(esm, axis=0)  #wyliczenie sredniej wartosci wsparcia
        prediction_array = np.argmax(average, axis=1)  #wskazanie etykiet

        return self.classes_[prediction_array] #zwrocenie predykcji 
