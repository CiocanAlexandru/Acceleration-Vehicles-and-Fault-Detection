import common_library
class SVN:
    ## Trebuie un constructor in care sa specific bach size  numarul de neuroni si o instanta de Extract_Fueatures_Augmentation pentru fiecare  
    def __init__(self,Features_Exel=None):
        self.Features_Exel=Features_Exel
        if Features_Exel!=None:
            self.data=Features_Exel.Get_Instances()
            self.row=len(self.data[0][0])
            self.colum=len(self.data[0][0][0])
        else:
            self.data=None
            self.row=None
            self.colum=None
    def Model(self):
        return 0
    

    def Training(self):
        return 0
    
    def Nk_Fold_Traning(self):
        return 0
    
    def Tes(self):
        return 0
    
'''
Ideie 

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss

# Presupunem că aveți o matrice de caracteristici (100, 40, 324) și un vector de etichete (100, 3)
features_matrix = np.random.rand(100, 40, 324) ///  futures insa trebuie sa fac asa iau datele le randomize si le extrag  sefapar                                       
labels_vector = np.array([                     ///  data=np.rand(data)
    ["SUV", "Toyota", "good"],                 ///  le extrag dupa  data[1] si data[0]
    ["Sedan", "Honda", "bad"],
    # Adăugați mai multe exemple aici
])

# Transformați etichetele într-un format adecvat pentru clasificare multi-etichetă
mlb = MultiLabelBinarizer()
transformed_labels = mlb.fit_transform(labels_vector)

# Separați setul de date într-un set de antrenament și unul de testare
X_train, X_test, y_train, y_test = train_test_split(features_matrix, transformed_labels, test_size=0.2, random_state=42)

# Creați un model SVM pentru clasificare multi-etichetă
svm_classifier = SVC(kernel='linear')

# Antrenați modelul pe setul de antrenament
svm_classifier.fit(X_train, y_train)

# Faceți predicții pe setul de testare
predictions = svm_classifier.predict(X_test)

# Calculați acuratețea modelului și pierderea Hamming
accuracy = accuracy_score(y_test, predictions)
hamming = hamming_loss(y_test, predictions)

print("Acuratețea modelului SVM pentru clasificare multi-etichetă:", accuracy)
print("Pierderea Hamming:", hamming)
'''