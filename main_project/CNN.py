import common_library
class CNN:
    ## Trebuie un constructor in care sa specific bach size  numarul de neuroni si o instanta de Extract_Fueatures_Augmentation pentru fiecare  
    def __init__(self,Features_Exel=None):
        self.Features_Exel=Features_Exel
        if Features_Exel!=None:
            self.data=Features_Exel.Get_Instances()
            self.row=len(self.data[0][0])
            self.colum=len(self.data[0][0][0])
    def Model(self):
        return 0
    
    def Training(self):
        return 0
    
    def Nk_Fold_Traning(self):
        return 0
    
    def Tes(self):
        return 0
    

'''
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

# Construiți modelul CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(num_features, feature_dim)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compilează modelul
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Antrenați modelul pe setul de antrenament
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Faceți predicții pe setul de testare
predictions = model.predict(X_test)

# Calculați acuratețea modelului
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
print("Acuratețea modelului CNN:", accuracy)
'''
    