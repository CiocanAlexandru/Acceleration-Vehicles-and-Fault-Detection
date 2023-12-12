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
          print("Shape features: ",self.features.shape)
          flattened_features = common_library.np.array(self.features)

          num_samples, num_features, num_values = flattened_features.shape

          flattened_features = flattened_features.reshape((num_samples * num_features, num_values))

          adjusted_transformed_labels = common_library.np.repeat(self.transformed_labels, num_features, axis=0)

          X_train, X_test, y_train, y_test = common_library.train_test_split(flattened_features[:300], adjusted_transformed_labels[:300], test_size=0.3)
          svm_model = common_library.OneVsRestClassifier(common_library.SVC())
          param_grid = {
              'estimator__C': [0.1, 1, 10, 100],        
              'estimator__kernel': ['linear', 'rbf'],   
              'estimator__gamma': [1, 0.1, 0.01, 0.001, 0.0001] 
          }
          grid_search = common_library.GridSearchCV(svm_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1, error_score='raise')
          
          grid_search.fit(X_train, y_train)
          print("Shape features: ",self.features.shape)
          best_params = grid_search.best_params_
          print("Best Hyperparameters:", best_params)
          self.C= grid_search.best_params_['estimator__C']
          self.Kernel = grid_search.best_params_['estimator__kernel']
          self.Gamma = grid_search.best_params_['estimator__gamma']
          self.Vizualize_GridShearch(X_train,y_train)
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
           self.Vizualize_GridShearch(X_train,y_train)
            
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
    def Vizualize_GridShearch(self,X_train,y_train):
        #def Vizualize_GridShearch(gamma_values,C_values,cv_results):
        param_grid = {'estimator__C': [1, 10, 100], 'estimator__gamma': [0.1, 0.01, 0.001,0.0001]}
        svm =  common_library.OneVsRestClassifier(common_library.SVC(kernel=self.Kernel))
        grid_search =common_library.GridSearchCV(svm, param_grid, cv=5)
        grid_search.fit(X_train[:500], y_train[:500])
        
        # Extrage rezultatele
        cv_results = grid_search.cv_results_
        
        # Vizualizare diagramă de contur pentru parametrul gamma
        common_library.plt.figure(figsize=(10, 8))
        
        # Creare grid pentru gamma și C
        gamma_values = common_library.np.array(param_grid['estimator__gamma'])
        C_values = common_library.np.array(param_grid['estimator__C'])
        gamma_values, C_values = common_library.np.meshgrid(gamma_values, C_values)
        
        # Plasare diagramă de contur în funcție de performanța modelului pentru fiecare combinație gamma-C
        self.diagrams.Vizualize_GridShearch(gamma_values,C_values,cv_results['mean_test_score'])                              
        return 0
        
    def Nk_Fold_Traning(self,features_extraction_method,cycles_nkfold=False):
        if cycles_nkfold==False:
            if self.features[0].shape[0]==40:
                flattened_features = common_library.np.array(self.features)

                num_samples, num_features, num_values = flattened_features.shape

                flattened_features = flattened_features.reshape((num_samples * num_features, num_values))

                adjusted_transformed_labels = common_library.np.repeat(self.transformed_labels, num_features, axis=0)
                self.model=self.Model()
                stratified_kfold = common_library.MultiLabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                cross_val_scores = common_library.cross_val_score(self.model, flattened_features, adjusted_transformed_labels, cv=stratified_kfold, scoring='accuracy')
                print("Cross-Validation Scores:", cross_val_scores)
                print("Mean Accuracy:", common_library.np.mean(cross_val_scores))
                self.accuracy=common_library.np.mean(cross_val_scores)
            else:
                print()

        if cycles_nkfold==True:
            if self.features[0].shape[0]==40:
                print()
            else:
                print()
        
        return 0
    def Test(self):
         print(f"Suport Vector Machine acuracy is :{self.accuracy}")
         return 0
    
