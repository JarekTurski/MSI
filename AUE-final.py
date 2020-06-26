from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaseEnsemble
import sklearn.utils.validation as suvalid #training utilities module
import numpy as np

# metody partial_fit oraz predict sa uzywane w TestThenTrain.process

class AUEFinal(ClassifierMixin, BaseEnsemble):

    #konstruktor
    def __init__(self, base_estimator, epsilon = 0.00000001):
        self.base_estimator = base_estimator
        self.epsilon = epsilon

    def check_fitted(self,clf): 
        return hasattr(clf, "ensembles_")

    def partial_fit(self, X, y, classes):
        X, y = suvalid.check_X_y(X, y)  #sprawdzanie poprawnosci danych wejsciowych
        print(X)

        #algorytm AUE
        #p_c = 
        #tworzenie tablicy ensemble_ dla k klasyfikatorow o najwiekszej wadze
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
        #obliczanie MSEr
        #mser = np.sum(p_c * np.power((1 - p_c), 2))

        #wyuczyc Ci na X 
         
        #obliczyc MSE dla Ci, walidacja krzyzowa na X

        #obliczenie wagi
        #w_i = 1 / (msei + self.epsilon)

        #petla do obliczania wag pozostalych klasyfikatorow z komitetu

        #petla: update k klasyfikatorow o najwiekszej wadze


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
        
    