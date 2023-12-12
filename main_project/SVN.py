import common_library 
common_library.warnings.filterwarnings("ignore")
class SVN:
    def __init__(self,class_index,encoded_data=None,diagrams=None,features_name=None):
        self.encoded_data=encoded_data
        self.diagrams=diagrams
        self.class_index=class_index
        self.features=common_library.np.array([i[0] for i in  self.encoded_data])
        self.transformed_labels=common_library.np.array([i[1] for i in  self.encoded_data])
        print(self.transformed_labels[0])
        print(class_index)
        print("SVN model initialiezed ")
        self.features_name=features_name
        self.name="SVN"
        self.C=None
        self.Gamma=None
        self.Kernel=None
        self.model=None
        self.GridShearch()
    def GridShearch(self):
        if self.features[0].shape[0]==40:
          flattened_features = common_library.np.array(self.features)

          num_samples, num_features, num_values = flattened_features.shape

          flattened_features = flattened_features.reshape((num_samples * num_features, num_values))

          adjusted_transformed_labels = common_library.np.repeat(self.transformed_labels, num_features, axis=0)

          X_train, X_test, y_train, y_test = common_library.train_test_split(flattened_features[:100], adjusted_transformed_labels[:100], test_size=0.3)
          svm_model = common_library.OneVsRestClassifier(common_library.SVC())
          param_grid = {
              'estimator__C': [0.1, 1, 10, 100],        
              'estimator__kernel': ['linear', 'rbf'],   
              'estimator__gamma': [1, 0.1, 0.01, 0.001, 0.0001] 
          }
          grid_search = common_library.GridSearchCV(svm_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1, error_score='raise')
          
          grid_search.fit(X_train, y_train)
          
          best_params = grid_search.best_params_
          print("Best Hyperparameters:", best_params)
          self.C= grid_search.best_params_['estimator__C']
          self.Kernel = grid_search.best_params_['estimator__kernel']
          self.Gamma = grid_search.best_params_['estimator__gamma']
        else:
           #FFT or PSD
           
           X_train, X_test, y_train, y_test=common_library.train_test_split(self.features[0:500],self.transformed_labels[0:500],test_size=0.3)
           svm_model = common_library.OneVsRestClassifier(common_library.SVC())
           param_grid = {
           'estimator__C': [0.1, 1, 10, 100],         
           'estimator__kernel': ['linear', 'rbf'],   
           'estimator__gamma': [1, 0.1, 0.01, 0.001,0.0001] 
           }
           grid_search = common_library.GridSearchCV(svm_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)

           grid_search.fit(X_train, y_train)

           best_params = grid_search.best_params_

           print("Best Hyperparameters:", best_params)
           self.C= grid_search.best_params_['estimator__C']
           self.Kernel = grid_search.best_params_['estimator__kernel']
           self.Gamma = grid_search.best_params_['estimator__gamma']
    def Model(self):
        model=common_library.OneVsRestClassifier(common_library.SVC(C=self.C, kernel=self.Kernel, gamma=self.Gamma))
        return model
    def Training(self,features_extraction_method):
        title='Average Confusion Matrix SVM whit '+features_extraction_method+' kerne='+str(self.Kernel)+' gamma='+str(self.Gamma)+' C='+str(self.C)
        if self.features[0].shape[0]==40:
            #Train_confusion_matrix(content,title,test)
            flattened_features = common_library.np.array(self.features)

            num_samples, num_features, num_values = flattened_features.shape

            flattened_features = flattened_features.reshape((num_samples * num_features, num_values))

            adjusted_transformed_labels = common_library.np.repeat(self.transformed_labels, num_features, axis=0)

            X_train, X_test, y_train, y_test = common_library.train_test_split(flattened_features[:100], adjusted_transformed_labels[:100], test_size=0.2)
            self.model=self.Model()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            self.accuracy = common_library.accuracy_score(y_test, y_pred)
           
            conf_matrix = common_library.multilabel_confusion_matrix(y_test, y_pred)
    
           
            average_conf_matrix=common_library.np.mean(conf_matrix,axis=0)

            
            self.diagrams.Train_confusion_matrix(average_conf_matrix,title,y_test)
        else:
            
            X_train, X_test, y_train, y_test = common_library.train_test_split(self.features[:100], self.transformed_labels[:100], test_size=0.2)
            self.model=self.Model()
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            self.accuracy = common_library.accuracy_score(y_test, y_pred)
            
            conf_matrix = common_library.multilabel_confusion_matrix(y_test, y_pred)
    
            
            average_conf_matrix=common_library.np.mean(conf_matrix,axis=0)

            
            self.diagrams.Train_confusion_matrix(average_conf_matrix,title,y_test)
                
        return 0
    def Nk_Fold_Traning(self,features_extraction_method):
        return 0
    def Test(self):
         print(f"Suport Vector Machine acuracy is :{self.accuracy}")
         return 0
    
