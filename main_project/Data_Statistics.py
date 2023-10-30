# O clasa care contine functiile urmatoare
# functie de afisare instante in consola pentru fiecare sau  o categorie  
# functie care deseneaza o distributie a bazei de date  (graf cu culori pentru masini  cu defecte si nu pe categorii )
## Atentie !! Aici ar fi de preferat sa folosesti dictionarele din proiectul initial  ca sa fie mai usor de facut 
import common_library

class Data_Statistic:
    def __init__(self,data=None):
        if(data!=None):
          self.data_frame=data.Get_DataFrame()
          self.exel=data  ## Data Manipulation object
        else:
           print("For models Diagrams") 
    def Number_of_Instance(self,info):
        if (self.data!=None):
            print(f" Number of instances {info}= {self.data[info].count()}")
        else:
            print("Operation Not working")
    def Dsitribution_Data_Base(self,info):
        if (self.data!=None):
            print(common_library.sns.displot(self.data_frame,x=info))
        else:
            print("Operation not working")
    def Data_Frame_Diagram_WAV(self,data_frame,sample_rate):
        if (self.data!=None):
            print(common_library.librosa.waveplot(data_frame,sr=sample_rate))
        else:
            print("Operation not working")
    def Accuracy_Model(history,function,Nkfold=False,function_number=False):
      print(function)
      if Nkfold==False:
        if function_number==False :
         common_library.plot.style('seaborn')
         common_library.plt.plot(history.history['accuracy'])
         common_library.plt.plot(history.history['val_accuracy'])
         common_library.plt.title('model accuracy normal training')
         common_library.plt.ylabel('accuracy')
         common_library.plt.xlabel('epoch')
         common_library.plt.legend(['train', 'test'], loc='upper left')
         common_library.plt.show()
        else:
            print("More loss functions")
            common_library.plot.style('seaborn')
            for i in history:
              common_library.plt.plot(i.history['accuracy'])
              common_library.plt.plot(i.history['val_accuracy'])
              common_library.plt.title('model accuracy normal training')
              common_library.plt.ylabel('accuracy')
              common_library.plt.xlabel('epoch')
              common_library.plt.legend(['train', 'test'], loc='upper left')
            common_library.plt.show()
            ## Doar digrama de validare
      else:
        if function_number==False:
           common_library.plot.style('seaborn')
           for i in history:
              common_library.plt.plot(i.history['accuracy'])
              common_library.plt.plot(i.history['val_accuracy'])
              common_library.plt.title('model accuracy kfold')
              common_library.plt.ylabel('accuracy')
              common_library.plt.xlabel('epoch')
              common_library.plt.legend(['train', 'test'], loc='upper left')
           common_library.plt.show()
        else:
           print("More functionfor nkfold function")
           common_library.plot.style('seaborn')
           for i in history:
              common_library.plt.plot(i.history['accuracy'])
              common_library.plt.plot(i.history['val_accuracy'])
              common_library.plt.title('model accuracy kfold')
              common_library.plt.ylabel('accuracy')
              common_library.plt.xlabel('epoch')
              common_library.plt.legend(['train', 'test'], loc='upper left')
           common_library.plt.show()
        return 0
    def Loss_Digrams(history,function,Nkfold=False,function_number=False):
      print(function)
      if Nkfold==False:
        if function_number==False :
         common_library.plot.style('seaborn')
         common_library.plt.plot(history.history['loss'])
         common_library.plt.plot(history.history['val_loss'])
         common_library.plt.title('model accuracy')
         common_library.plt.ylabel('loss')
         common_library.plt.xlabel('epoch')
         common_library.plt.legend(['train', 'test'], loc='upper left')
         common_library.plt.show()
        else:
            print("More loss functions")
            common_library.plot.style('seaborn')
            for i in history:
              common_library.plt.plot(i.history['loss'])
              common_library.plt.plot(i.history['val_loss'])
              common_library.plt.title('model accuracy normal training')
              common_library.plt.ylabel('loss')
              common_library.plt.xlabel('epoch')
              common_library.plt.legend(['train', 'test'], loc='upper left')
            common_library.plt.show()
      else:
        if function_number==False:
            print("Only one function for nkfold")
            common_library.plot.style('seaborn')
            for i in history:
              common_library.plt.plot(i.history['loss'])
              common_library.plt.plot(i.history['val_loss'])
              common_library.plt.title('model accuracy normal training')
              common_library.plt.ylabel('loss')
              common_library.plt.xlabel('epoch')
              common_library.plt.legend(['train', 'test'], loc='upper left')
            common_library.plt.show()
        else:
           print("More functionfor nkfold function")
           common_library.plot.style('seaborn')
           for i in history:
              common_library.plt.plot(i.history['loss'])
              common_library.plt.plot(i.history['val_loss'])
              common_library.plt.title('model accuracy normal training')
              common_library.plt.ylabel('loss')
              common_library.plt.xlabel('epoch')
              common_library.plt.legend(['train', 'test'], loc='upper left')
           common_library.plt.show()
        return 0
        

