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
        self.GridShearch()
        #self.best_settings=self.GridShearch()
        #self.C=None
        #self.Gamma=None
        #self.Kernel='2'
    def GridShearch(self):
        if self.features[0].shape[0]==40:
           #MFCC
           
           print("Best Hyperparameters:", best_params)
        else:
           #FFT or PSD
           ##Inpartire set de date
           X_train, X_test, y_train, y_test=common_library.train_test_split(self.features,self.transformed_labels,test_size=3)
           svm_model = common_library.OneVsRestClassifier(common_library.SVC())
           param_grid = {
           'estimator__C': [0.1, 1, 10, 100],         # Regularization parameter
           'estimator__kernel': ['linear', 'rbf'],   # Kernel type
           'estimator__gamma': [1, 0.1, 0.01, 0.001, 0.0001] # Kernel coefficient for 'rbf' kernel
           }
           X_train=X_train[0:500]
           y_train=y_train[0:500]
           grid_search = common_library.GridSearchCV(svm_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
           grid_search.fit(X_train, y_train)
           best_params = grid_search.best_params_
           print("Best Hyperparameters:", best_params)
    
    def Model(self):
        return 0
    def Training(self):
        return 0
    def Test(self):
         return 0
    def Nk_Fold_Traning(self):
        return 0
    
