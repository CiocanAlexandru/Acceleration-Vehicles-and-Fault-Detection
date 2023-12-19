import common_library
import tensorflow as tf
from keras import backend as K
class FCNN:  
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
        self.model_name="FCNN"
    def Model(self):
        print(self.features[0].shape)
        num_classes=len(self.class_index)+1
        self.model = common_library.tf.keras.Sequential([common_library.tf.keras.layers.Flatten(input_shape=self.features[0].shape),  # Aplatizează intrarea
         common_library.tf.keras.layers.Dense(64, activation='relu'),
        common_library.tf.keras.layers.Dense(128, activation='linear'),  # Stratul ascuns cu 128 de neuroni și activare liniară
        common_library.tf.keras.layers.Dense(64, activation='relu'),  # Al doilea strat ascuns cu 64 de neuroni și activare ReLU
        common_library.tf.keras.layers.Dense(num_classes, activation='sigmoid')])
        return self.model  # Stratul de ieșire cu 3 neuroni și activare Softmax])
    def Training(self,number_loss=False):
        # def Accuracy_Diagrams(self,history,function,function_number=False,features_name=None,model_name=None):
        # def Loss_Diagrams(self,history,function,function_number=False,features_name=None,model_name=None):
        loss_functions=['categorical_crossentropy','binary_crossentropy']
        loss_function='binary_crossentropy'
        if number_loss==True:
           print("Normal training whit multi loss function")
           history_list=[]
           accuracy_list=[]
           for iloss_function in loss_functions:
                X_train, X_test, y_train, y_test = common_library.train_test_split(self.features, self.transformed_labels, test_size=0.3)
                self.model=self.Model()
                self.model.compile(optimizer='adam', loss=iloss_function, metrics=['binary_accuracy'])
                history=self.model.fit(X_train,y_train, epochs=100, batch_size=10, validation_split=0.3)
                history_list.append(history)
                accuracy_list.append(self.model.evaluate(X_test,y_test)[1])
           self.accuracy={loss_functions[i]:accuracy_list[i] for i in range(0,len(loss_functions))}
           print(self.accuracy)
           #self.accuracy=common_library.np.mean(accuracy_list,axis=0)
           self.diagrams.Accuracy_Diagrams(history_list,loss_functions,True,self.features_name,self.model_name,self.accuracy)
           self.diagrams.Loss_Diagrams(history_list,loss_functions,True,self.features_name,self.model_name)
        if number_loss==False:
           print("Normal Traing whit one loss function ")
           X_train, X_test, y_train, y_test = common_library.train_test_split(self.features, self.transformed_labels, test_size=0.3)
           self.model=self.Model()
           self.model.compile(optimizer='adam', loss=loss_function, metrics=['binary_accuracy'])
           history=self.model.fit(X_train,y_train, epochs=100, batch_size=10, validation_split=0.3)
           self.accuracy=self.model.evaluate(X_test,y_test)[1]
           self.diagrams.Accuracy_Diagrams(history,loss_function,False,self.features_name,self.model_name,self.accuracy)
           self.diagrams.Loss_Diagrams(history,loss_function,False,self.features_name,self.model_name)
        return 0
    def Nk_Fold_Traning(self,number_loss=False,cycles_nkfold=False):
        #def Accuracy_Diagrams_Nkfold(self,history,function,function_number=False,features_name=None,model_name=None,cycles_nkfold=False,accuracy=None):
        #def Loss_Diagrams_Nkfold(self,history,function,function_number=False,features_name=None,model_name=None,cycles_nkfold=False,accuracy=None):
        loss_functions=['categorical_crossentropy','binary_crossentropy']
        loss_function='binary_crossentropy'
        if number_loss==False and cycles_nkfold==False:     ## Diagrams for every fold  each 
            print("Nkfold whit one lost function and no cycles")
            n_splits=5
            skf = common_library.KFold(n_splits=n_splits, shuffle=True)
            history_list=[]
            accuracy_list=[]
            for train_index, val_index in skf.split(self.features, self.transformed_labels):
              X_train, X_val = self.features[train_index], self.features[val_index]
              y_train, y_val = self.transformed_labels[train_index], self.transformed_labels[val_index]
              model = self.Model()
              model.compile(optimizer='adam', loss=loss_function, metrics=['binary_accuracy'])
              history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
              accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
              history_list.append(history)
              accuracy_list.append(accuracy)
            self.accuracy=common_library.np.mean(accuracy_list,axis=0)
            self.diagrams.Accuracy_Diagrams_Nkfold(history_list,loss_function,False,self.features_name,self.model_name,False,self.accuracy)
            self.diagrams.Loss_Diagrams_Nkfold(history_list,loss_function,False,self.features_name,self.model_name,False)
        if number_loss==True and cycles_nkfold==False:      ## Diagram for every fold  each
            print("Nkfold whit multi lost function and no cycles")
            n_splits=5
            skf = common_library.KFold(n_splits=n_splits, shuffle=True)
            history_list=[]
            accuracy_list=[]
            for iloss_function in loss_functions:
                fold_accuracies_for_loss = []
                historys_for_loss = []
                for train_index, val_index in skf.split(self.features, self.transformed_labels):
                 X_train, X_val = self.features[train_index], self.features[val_index]
                 y_train, y_val = self.transformed_labels[train_index], self.transformed_labels[val_index]
                 model = self.Model()
                 model.compile(optimizer='adam', loss=iloss_function, metrics=['binary_accuracy'])
                 history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
                 accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
                 fold_accuracies_for_loss.append(accuracy)
                 historys_for_loss.append(history) 
                accuracy_list.append(fold_accuracies_for_loss)
                history_list.append(historys_for_loss)
            self.accuracy={loss_functions[i]:common_library.np.mean(accuracy_list[i]) for i in range(0,len(accuracy_list))}
            self.diagrams.Accuracy_Diagrams_Nkfold(history_list,loss_functions,True,self.features_name,self.model_name,False,self.accuracy)
            self.diagrams.Loss_Diagrams_Nkfold(history_list,loss_functions,True,self.features_name,self.model_name,False)
        if number_loss==False and cycles_nkfold==True:      ## Diagrams for more cycles each [1,3,5,7]
            print("Nkfold whit one lost function and cycles")
            cycles=[1,3,5,7]
            n_splits=5
            skf = common_library.KFold(n_splits=n_splits, shuffle=True)
            history_mean=[]
            accuracy_mean=[]
            for cycle in cycles:
                print("----------------------------")
                print(f"Number of the cycle {cycle}")
                print("-----------------------------")
                history_list=[]
                accuracy_list=[]
                for i in range(cycle):
                 print("----------------------------")
                 print(f"Number kfold {i} for cycle {cycle}")
                 print("-----------------------------")
                 for train_index, val_index in skf.split(self.features, self.transformed_labels):
                  X_train, X_val = self.features[train_index], self.features[val_index]
                  y_train, y_val = self.transformed_labels[train_index], self.transformed_labels[val_index]
                  model = self.Model()
                  model.compile(optimizer='adam', loss=loss_function, metrics=['binary_accuracy'])
                  history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
                  accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
                  history_list.append(history)
                  accuracy_list.append(accuracy)
                history_mean.append({
                 'loss': common_library.np.mean([h.history['loss'] for h in history_list], axis=0),
                 'val_loss': common_library.np.mean([h.history['val_loss'] for h in history_list], axis=0),
                 'binary_accuracy': common_library.np.mean([h.history['binary_accuracy'] for h in history_list], axis=0),
                 'val_binary_accuracy': common_library.np.mean([h.history['val_binary_accuracy'] for h in history_list], axis=0)
                 })
                accuracy_mean.append(common_library.np.mean(accuracy_list,axis=0))
                self.accuracy=common_library.np.mean(accuracy_mean,axis=0)    
            self.diagrams.Accuracy_Diagrams_Nkfold(history_mean,loss_function,False,self.features_name,self.model_name,True,accuracy_mean,n_splits)
            self.diagrams.Loss_Diagrams_Nkfold(history_mean,loss_function,False,self.features_name,self.model_name,True,n_splits)
        if number_loss==True and cycles_nkfold==True:       ## Diagram for more cycles each [1,3,5,7]
            print("Nkfold whit multi lost function and cycles") 
            cycles=[1,3,5,7]
            n_splits=5
            skf = common_library.KFold(n_splits=n_splits, shuffle=True)
            history_all=[]
            accuracy_all=[]
            for iloss_function in loss_functions:
             print("-----------------------------")
             print(f"Function is {iloss_function}")
             print("-----------------------------")
             history_mean=[]
             accuracy_mean=[]
             for cycle in cycles:
                 print("----------------------------")
                 print(f"Number of the cycle {cycle}")
                 print("-----------------------------")
                 history_list=[]
                 accuracy_list=[]
                 for i in range(cycle):
                  print("----------------------------")
                  print(f"Number kfold {i} for cycle {cycle}")
                  print("-----------------------------")
                  for train_index, val_index in skf.split(self.features, self.transformed_labels):
                   X_train, X_val = self.features[train_index], self.features[val_index]
                   y_train, y_val = self.transformed_labels[train_index], self.transformed_labels[val_index]
                   model = self.Model()
                   model.compile(optimizer='adam', loss=iloss_function, metrics=['binary_accuracy'])
                   history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
                   accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
                   history_list.append(history)
                   accuracy_list.append(accuracy)
                 history_mean.append({
                  'loss': common_library.np.mean([h.history['loss'] for h in history_list], axis=0),
                  'val_loss': common_library.np.mean([h.history['val_loss'] for h in history_list], axis=0),
                  'binary_accuracy': common_library.np.mean([h.history['binary_accuracy'] for h in history_list], axis=0),
                  'val_binary_accuracy': common_library.np.mean([h.history['val_binary_accuracy'] for h in history_list], axis=0)
                  })
                 accuracy_mean.append(common_library.np.mean(accuracy_list,axis=0))
             history_all.append(history_mean)
             accuracy_all.append(accuracy_mean)
            print(f"Elemente histori all{len(history_all)}")
            print(f"Eelemente in fisrt element{len(history_all[0])} and second {len(history_all[1])}")
            print(f"Elemente acuracy all{len(accuracy_all)}")
            print(f"Eelemente in fisrt element{len(accuracy_all[0])} and second {len(accuracy_all[1])}")
            self.diagrams.Accuracy_Diagrams_Nkfold(history_all,loss_functions,True,self.features_name,self.model_name,True,accuracy_all,n_splits)
            self.diagrams.Loss_Diagrams_Nkfold(history_all,loss_functions,True,self.features_name,self.model_name,True,n_splits)
        return 0
    def Test(self):
         print(f"Neuronal model acuracy is :{self.accuracy}")
         return 0





'''
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
    
    def Nk_Fold_Traning(self,number_loss=False,cycles_nkfold=False):
        path="./Models/"
        if number_loss==False and cycles_nkfold==False:
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
            # def Accuracy_Model_Nkfold(self,history,function,function_number=False,features_name=None,model_name=None,cycles_nkfold=False):
            self.diagrams.Accuracy_Model(historys,loss_function,True,False,self.features_name,self.name)
            self.diagrams.Loss_Digrams(historys,loss_function,True,False,self.features_name,self.name)
        if number_loss==True and cycles_nkfold==False:
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
        if number_loss==False and cycles_nkfold==True:
            print("One lost function multiple cycles")
        if number_loss==True and cycles_nkfold==True:
           print("More lost function multiple cycles")
        return 0
'''