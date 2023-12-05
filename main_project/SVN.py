import common_library
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
    def Model(self):
        return 0
    def Training(self,number_loss=False):
        return 0
    def Test(self):
         return 0
    
    def Nk_Fold_Traning(self,number_loss=False,cycles_nkfold=False):
        return 0
    
