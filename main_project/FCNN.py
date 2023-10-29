import common_library
class FCNN:
    ## Trebuie un constructor in care sa specific bach size  numarul de neuroni si o instanta de Extract_Fueatures_Augmentation pentru fiecare  
    def __init__(self,class_index,encoded_data=None,diagrams=None):
        self.encoded_data=encoded_data
        self.diagrams=diagrams
        self.class_index=class_index
        self.features=common_library.np.array([i[0] for i in  self.encoded_data])
        self.transformed_labels=common_library.np.array([i[1] for i in  self.encoded_data])
        print("FCNN model initialiezed ")
    def Model(self):
        self.model = common_library.tf.keras.Sequential([common_library.tf.keras.layers.Flatten(input_shape=(self.features, self.features.shape)),  # Aplatizează intrarea
        common_library.tf.keras.layers.Dense(128, activation='linear'),  # Stratul ascuns cu 128 de neuroni și activare liniară
        common_library.tf.keras.layers.Dense(64, activation='relu'),  # Al doilea strat ascuns cu 64 de neuroni și activare ReLU
        common_library.tf.keras.layers.Dense(len(self.class_index), activation='softmax')])
        return self.model  # Stratul de ieșire cu 3 neuroni și activare Softmax])
    def Training(self,number_loss=False):
        if number_loss==False:
          self.X_train, self.X_test, self.y_train, self.y_test = common_library.train_test_split(self.features, self.transformed_labels, test_size=0.2, random_state=42)
          loss_function='categorical_crossentropy'
          self.model=self.Model()
          self.model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
          history=self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, validation_split=0.2)
          print(history.history.keys())
          self.diagrams.Accuracy_Model(history,loss_function,False,False)
          self.diagrams.Loss_Digrams(history,loss_function,False,False)
        else:
          loss_function=['categorical_crossentropy','binary_crossentropy','sparse_categorical_crossentropy']
          self.diagrams.Accuracy_Model(history,loss_function,False,True)
          self.diagrams.Loss_Digrams(history,loss_function,False,True)
          print("Training whit more loss functions")
        return 0
    # def Accuracy_Model(history,function,Nkfold=False,function_number=False):
    def Test(self,number_loss=False,Nk_fold=False):
     if Nk_fold==False:
        if (number_loss==False):
         predictions = self.model.predict(self.X_test)
         # Calculați acuratețea modelului
         accuracy = common_library.accuracy_score(common_library.np.argmax(self.y_test, axis=1), common_library.np.argmax(predictions, axis=1))
         print("Acuratețea modelului neural:", accuracy)
        else:
          print("")
     else:
        print("")
        return 0
    
    def Nk_Fold_Traning(self,number_loss=False):
        if number_loss==False:
           print("yes")
        else:
           loss_function=['categorical_crossentropy','binary_crossentropy','sparse_categorical_crossentropy']
           print("no")
        return 0

'''
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

# Generați date de antrenament
num_samples = 100
num_features = 40
feature_dim = 324

# Generați o singură matrice de caracteristici cu dimensiunea (100, 40, 324)
features_matrix = np.random.rand(num_samples, num_features, feature_dim)

# Generați etichete pentru fiecare exemplu
labels_vector = np.array([
    ["SUV", "Toyota", "good"],
    ["Sedan", "Honda", "bad"],
    # Adăugați mai multe exemple aici
])

# Transformați etichetele într-un format adecvat pentru clasificare multi-etichetă
mlb = MultiLabelBinarizer()
transformed_labels = mlb.fit_transform(labels_vector)

# Separați setul de date într-un set de antrenament și unul de testare
X_train, X_test, y_train, y_test = train_test_split(features_matrix, transformed_labels, test_size=0.2, random_state=42)

# Construiți modelul rețelei neurale
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(num_features, feature_dim)),  # Aplatizează intrarea
    tf.keras.layers.Dense(128, activation='relu'),  # Stratul ascuns cu 128 de neuroni și activare ReLU
    tf.keras.layers.Dense(64, activation='relu'),  # Al doilea strat ascuns cu 64 de neuroni și activare ReLU
    tf.keras.layers.Dense(3, activation='softmax')  # Stratul de ieșire cu 3 neuroni și activare Softmax
])

# Compilează modelul
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Antrenați modelul pe setul de antrenament
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Faceți predicții pe setul de testare
predictions = model.predict(X_test)

# Calculați acuratețea modelului
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
print("Acuratețea modelului neural:", accuracy)
'''