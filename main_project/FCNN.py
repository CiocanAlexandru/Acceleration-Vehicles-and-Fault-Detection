import common_library
class FCNN:
    ## Trebuie un constructor in care sa specific bach size  numarul de neuroni si o instanta de Extract_Fueatures_Augmentation pentru fiecare  
    def __init__(self,data=None,encoded_data=None,diagrams=None):
        self.data=data
        self.encoded_data=encoded_data
        self.diagrams=diagrams
        return 0
    def Model(self):
        
        return 0
    
    def Training(self):
        return 0
    
    def Test(self):
        return 0
    
    def Nk_Fold_Traning(self):
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