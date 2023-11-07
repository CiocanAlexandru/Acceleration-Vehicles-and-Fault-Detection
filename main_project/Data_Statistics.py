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
    def Accuracy_Model(self,history,function,Nkfold=False,function_number=False,features_name=None,model_name=None):
      print(function)
      path="./Diagrams/"
      date=common_library.datetime.now().strftime('%Y-%m-%d %I-%M-%S %p')
      full_name_diagram=path+"Accuracy_"+model_name+"_"+date+"_"+features_name+".jpg"
      if Nkfold==False:
        if function_number==False :
         common_library.plt.style.use('seaborn')
         common_library.plt.plot(history.history['binary_accuracy'],label=function)
         common_library.plt.plot(history.history['val_binary_accuracy'],label=function)
         common_library.plt.title('model accuracy normal training whit one lost func and'+features_name)
         common_library.plt.ylabel('accuracy')
         common_library.plt.xlabel('epoch')
         common_library.plt.legend(['tr ain', 'test'], loc='upper left')
         common_library.plt.savefig(full_name_diagram,format='jpg')
         common_library.plt.show()
        else:
            print("More loss functions")
            common_library.plt.style.use('seaborn')
            k=0
            for i in history:
              common_library.plt.plot(i.history['binary_accuracy'],label="Training "+function[k])
              common_library.plt.plot(i.history['val_binary_accuracy'],label="Validation "+function[k])
              common_library.plt.title('model accuracy norm train whit multiple lost func and '+features_name)
              k+=1
            common_library.plt.ylabel('accuracy')
            common_library.plt.xlabel('epoch')
            common_library.plt.legend(loc='upper left')
            common_library.plt.savefig(full_name_diagram,format='jpg')
            common_library.plt.show()
            ## Doar digrama de validare
      else:
        if function_number==False:
           common_library.plt.style.use('seaborn')
           k=0
           labels=[]
           for i in history:
              label=("K-Fold="+str(k))
              common_library.plt.plot(i.history['val_binary_accuracy'],label=label)
              labels.append(label)
              k+=1
           common_library.plt.title('model accuracy kfold whit one lost func and '+features_name)
           common_library.plt.ylabel('accuracy')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(labels, loc='upper left')
           common_library.plt.savefig(full_name_diagram,format='jpg')
           common_library.plt.show()
        else:
           common_library.plt.style.use('seaborn')
           k=0
           labels=[]
           for i in history:
              label=(function[k])
              for j in range(0,len(i)):
               common_library.plt.plot(i[j].history['val_binary_accuracy'],label=label)
               labels.append(label)
              k+=1
           common_library.plt.title('model accuracy kfold whit mult lost func and '+features_name)
           common_library.plt.ylabel('accuracy')
           common_library.plt.legend(labels,loc='lower right')
           common_library.plt.savefig(full_name_diagram,format='jpg')
           common_library.plt.show()
           #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        return 0
    def Loss_Digrams(self,history,function,Nkfold=False,function_number=False,features_name=None,model_name=None):
      print(function)
      path="./Diagrams/"
      date=common_library.datetime.now().strftime('%Y-%m-%d %I-%M-%S %p')
      full_name_diagram=path+"Loss_"+model_name+"_"+date+"_"+features_name+".jpg"
      if Nkfold==False:
        if function_number==False :
         common_library.plt.style.use('seaborn')
         common_library.plt.plot(history.history['loss'],label=function)
         common_library.plt.plot(history.history['val_loss'],label=function)
         common_library.plt.title('model loss normal training whit one lost func and '+features_name)
         common_library.plt.ylabel('loss')
         common_library.plt.xlabel('epoch')
         common_library.plt.legend(['train', 'test'], loc='upper left')
         common_library.plt.savefig(full_name_diagram,format='jpg')
         common_library.plt.show()
        else:
            print("More loss functions")
            common_library.plt.style.use('seaborn')
            k=0
            for i in history:
              common_library.plt.plot(i.history['loss'],label="Training "+function[k])
              common_library.plt.plot(i.history['val_loss'],label="Validation "+function[k])
              k+=1
            common_library.plt.title('model loss normal traning whit mult lost func and '+features_name)
            common_library.plt.ylabel('loss')
            common_library.plt.xlabel('epoch')
            common_library.plt.legend(loc='upper left')
            common_library.plt.savefig(full_name_diagram,format='jpg')
            common_library.plt.show()
      else:
        if function_number==False:
            print("Only one function for nkfold")
            common_library.plt.style.use('seaborn')
            k=0
            labels=[]
            for i in history:
              label=("K-Fold="+str(k))
              common_library.plt.plot(i.history['loss'],label=label)
              common_library.plt.title('model loss')
              labels.append(label)
              k+=1
            common_library.plt.title('model loss kfold whit mult lost func and '+features_name)
            common_library.plt.ylabel('loss')
            common_library.plt.xlabel('epoch')
            common_library.plt.legend(labels, loc='upper left')
            common_library.plt.savefig(full_name_diagram,format='jpg')
            common_library.plt.show()
        else:
           common_library.plt.style.use('seaborn')
           k=0
           labels=[]
           for i in history:
              label=(function[k])
              for j in range(0,len(i)):
               common_library.plt.plot(i[j].history['val_loss'],label=label)
               labels.append(label)
              k+=1
           common_library.plt.title('model accuracy kfold whit mult lost func and '+features_name)
           common_library.plt.ylabel('loss')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(labels,loc='lower right')
           common_library.plt.savefig(full_name_diagram,format='jpg')
           common_library.plt.show()
        return 0
        

