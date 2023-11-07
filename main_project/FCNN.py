import common_library
class FCNN:
    ## Trebuie un constructor in care sa specific bach size  numarul de neuroni si o instanta de Extract_Fueatures_Augmentation pentru fiecare  
    def __init__(self,class_index,encoded_data=None,diagrams=None,features_name=None):
        self.encoded_data=encoded_data
        self.diagrams=diagrams
        self.class_index=class_index
        self.features=common_library.np.array([i[0] for i in  self.encoded_data])
        self.transformed_labels=common_library.np.array([i[1] for i in  self.encoded_data])
        print(self.transformed_labels[0])
        print(class_index)
        print("FCNN model initialiezed ")
        self.features_name=features_name
        self.name="FCNN"
    def Model(self):
        print(self.features[0].shape)
        self.model = common_library.tf.keras.Sequential([common_library.tf.keras.layers.Flatten(input_shape=self.features[0].shape),  # Aplatizează intrarea
         common_library.tf.keras.layers.Dense(64, activation='relu'),
        common_library.tf.keras.layers.Dense(128, activation='linear'),  # Stratul ascuns cu 128 de neuroni și activare liniară
        common_library.tf.keras.layers.Dense(64, activation='relu'),  # Al doilea strat ascuns cu 64 de neuroni și activare ReLU
        common_library.tf.keras.layers.Dense(len(self.class_index)+1, activation='sigmoid')])
        return self.model  # Stratul de ieșire cu 3 neuroni și activare Softmax])
    def Training(self,number_loss=False):
        path="./Models/"
        if number_loss==False:
          model_name=self.name+"_"+self.features_name+"_"
          print("Training FCNN normal traning whit only one lost function")
          X_train, X_test, y_train, y_test = common_library.train_test_split(self.features, self.transformed_labels, test_size=0.5, random_state=52)
          print(X_train.shape)
          print(X_test.shape)
          print(y_test.shape)
          print(y_train.shape)
          loss_function='binary_crossentropy'
          self.model=self.Model()
          self.model.compile(optimizer='adam', loss=loss_function, metrics=['binary_accuracy'])
          history=self.model.fit(X_train,y_train, epochs=100, batch_size=10, validation_split=0.5)
         # Calculați acuratețea modelului
         # threshold = 0.5
         # binary_predictions = (predictions >= threshold).astype(int)
         # correct_predictions = (binary_predictions == y_test).sum()
         # total_predictions = len(y_test)
         # self.accuracy=correct_predictions / total_predictions
          predictions=self.model.predict(X_test)

          #prediction_x=common_library.np.argmax(prediction, axis = 1)
          #prediction_y=common_library.np.argmax(y_test,axis = 1) 
          #total_number=len(prediction_x)
          #corect_number=0
          #prediction_indices = common_library.np.argmax(predictions, axis=1)
          #y_true = common_library.np.argmax(y_test, axis=1)
          print("---------------------")
          print(common_library.accuracy_score(common_library.np.argmax(y_test, axis=1), common_library.np.argmax(predictions, axis=1)))
          print("-----------------")
          #print(corect_number)
          #print(total_number)
          #print(corect_number/total_number)
          self.accuracy=self.model.evaluate(X_test,y_test)[1]
          #self.accuracy = common_library.accuracy_score(common_library.np.argmax(y_test, axis=1), common_library.np.argmax(predictions, axis=1))
          #binary_predictions = (predictions > 0.5).astype(int)
          #correct_predictions = (binary_predictions == y_test).sum()
          #total_predictions = len(y_test)
          #self.accuracy = correct_predictions / total_predictions
         # predictions = self.model.predict(X_test)
         # self.accuracy = common_library.accuracy_score(common_library.np.argmax(y_test, axis=1), common_library.np.argmax(predictions, axis=1))
          model_name+=loss_function
          model_name+=".h5"
          self.model.save(path+"normal_train_"+model_name)
          self.diagrams.Accuracy_Model(history,loss_function,False,False,self.features_name,self.name)
          self.diagrams.Loss_Digrams(history,loss_function,False,False,self.features_name,self.name)
        else:
          print("Training FCNN normal traning whit only one more lost function")
          loss_function=['categorical_crossentropy','binary_crossentropy']
          historys=[]
          accuracys=[]
          for i in loss_function:
             model_name=self.name+"_"+self.features_name+"_"
             print(i)
             X_train, X_test, y_train, y_test = common_library.train_test_split(self.features, self.transformed_labels, test_size=0.2, random_state=42)
             self.model=self.Model()
             self.model.compile(optimizer='adam', loss=i, metrics=['binary_accuracy'])
             history=self.model.fit(X_train,y_train, epochs=100, batch_size=32, validation_split=0.5)
             historys.append(history)
             predictions = self.model.predict(X_test)
             accuracy = common_library.accuracy_score(common_library.np.argmax(y_test, axis=1), common_library.np.argmax(predictions, axis=1))
             accuracys.append(accuracy)
             model_name+=i
             model_name+=".h5"
             self.model.save(path+"normal_train_multiLossFunc_"+model_name)
          self.accuracy=common_library.np.mean(accuracys)
          self.diagrams.Accuracy_Model(historys,loss_function,False,True,self.features_name,self.name)
          self.diagrams.Loss_Digrams(historys,loss_function,False,True,self.features_name,self.name)
          print("Training whit more loss functions")
        return 0
    # def Accuracy_Model(history,function,Nkfold=False,function_number=False):
    def Test(self):
         print("Acuratețea modelului neural:", self.accuracy)
         return 0
    
    def Nk_Fold_Traning(self,number_loss=False):
        path="./Models/"
        if number_loss==False:
            print("Training nkfold only whit one lost function")
            n_splits=5
            loss_function='binary_crossentropy'
            skf = common_library.KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_accuracies = []
            historys=[]
            models=[]
            k=1
            for train_index, val_index in skf.split(self.features, self.transformed_labels):
              print(f"N-fold={k}")
              model_name=self.name+"_"+self.features_name+"_"
              X_train, X_val = self.features[train_index], self.features[val_index]
              y_train, y_val = self.transformed_labels[train_index], self.transformed_labels[val_index]
              model = self.Model()
              model.compile(optimizer='adam', loss=loss_function, metrics=['binary_accuracy'])
              history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
              accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
              fold_accuracies.append(accuracy)
              historys.append(history)
              models.append(model)
              model_name=loss_function
              model_name+=".h5"
              model.save(path+"kFold_train_one_lossFunc_"+"k="+str(k)+"_"+model_name)
              k+=1
            self.accuracy = common_library.np.mean(fold_accuracies)
            #def Accuracy_Model(self,history,function,Nkfold=False,function_number=False):
            self.diagrams.Accuracy_Model(historys,loss_function,True,False,self.features_name,self.name)
            self.diagrams.Loss_Digrams(historys,loss_function,True,False,self.features_name,self.name)
        else:
           loss_function=['categorical_crossentropy','binary_crossentropy']
           n_splits = 10
           skf = common_library.KFold(n_splits=n_splits, shuffle=True, random_state=42)
           fold_accuracies = []
           historys = []
           models = []
           for loss_function in loss_function:
             fold_accuracies_for_loss = []
             historys_for_loss = []
             k=1
             for train_index, val_index in skf.split(self.features, self.transformed_labels):
               X_train, X_val = self.features[train_index], self.features[val_index]
               y_train, y_val = self.transformed_labels[train_index], self.transformed_labels[val_index]
               model_name=self.name+"_"+self.features_name+"_"
               model_name+=loss_function
               model_name+=".h5"
               model = self.Model()
               model.compile(optimizer='adam', loss=loss_function, metrics=['binary_accuracy'])
               history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
               accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
               fold_accuracies_for_loss.append(accuracy)
               fold_accuracies_for_loss.append(accuracy)
               historys_for_loss.append(history) 
               model.save(path+"kFold_train_multi_lossFunc_"+"k="+str(k)+"_"+model_name)
               k+=1
           fold_accuracies.append(fold_accuracies_for_loss)
           historys.append(historys_for_loss)
           self.accuracy= [common_library.np.mean(accuracies) for accuracies in fold_accuracies]
           self.diagrams.Accuracy_Model(historys,loss_function,True,True,self.features_name,self.name)
           self.diagrams.Loss_Digrams(historys,loss_function,True,True,self.features_name,self.name)
           print("no")
        return 0

