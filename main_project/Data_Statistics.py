# O clasa care contine functiile urmatoare
# functie de afisare instante in consola pentru fiecare sau  o categorie  
# functie care deseneaza o distributie a bazei de date  (graf cu culori pentru masini  cu defecte si nu pe categorii )
## Atentie !! Aici ar fi de preferat sa folosesti dictionarele din proiectul initial  ca sa fie mai usor de facut 
import common_library

class Data_Statistic:
    def __init__(self,data=None):
        if(data!=None):
          self.data_frame=data.Get_DataFrame()
          self.data=data  ## Data Manipulation object
        else:
           print("For models Diagrams") 
    def Number_of_Instance(self,info):
        if (self.data!=None):
            print(f"Number of instances {info}= {self.data_frame[info].count()}")
            print("Now for every instance in particular base on classes:")
            unique_class={i:0 for i in self.data_frame[info]}
            for i in self.data_frame[info]:
               unique_class[i]+=1
            for key,value in unique_class.items():
               print(f"For key= {key} we have: {value} of instances")
        else:
            print("Operation Not working")
    def Dsitribution_Data_Base(self,info):
        if (self.data!=None):
            common_library.plt.style.use('seaborn')
            common_library.sns.displot(self.data_frame,x=info)
            common_library.plt.title('Distribution Plot '+info)
            common_library.plt.xticks(rotation=45)
            common_library.plt.subplots_adjust(bottom=0.2)
            common_library.plt.savefig("./DiagramsWav/Distribution dataset by "+info+".jpg",format='jpg')
            common_library.plt.show()
            
        else:
            print("Operation not working")
    def Data_Frame_Diagram_WAV(self,data_frame,sample_rate,file_name,modified=None):
        if (self.data!=None):
            if (modified==None):
             common_library.librosa.display.waveshow(data_frame, sr=sample_rate)
             common_library.plt.title('Waveform for '+file_name)
             common_library.plt.xlabel('Time (s)')
             common_library.plt.ylabel('Amplitude')
             common_library.plt.show()
            else:
             common_library.librosa.display.waveshow(data_frame, sr=sample_rate)
             common_library.plt.title('Waveform for '+file_name+' modified')
             common_library.plt.xlabel('Time (s)')
             common_library.plt.ylabel('Amplitude')
             common_library.plt.show()  
        else:
            print("Operation not working")
    def Wav_Frame(self,audio_data=None,sample_rate=None,name=None):
        if (self.data!=None):
           common_library.librosa.display.waveshow(audio_data, sr=sample_rate)
           common_library.plt.title(name)
           common_library.plt.xlabel('Time (s)')
           common_library.plt.ylabel('Amplitude')
           common_library.plt.savefig('./DiagramsWav/'+name+".jpg",format='jpg')
           common_library.plt.show()
        else:
           print("Operation not working!!")
    def Accuracy_Diagrams(self,history,function,function_number=False,features_name=None,model_name=None,accuracy=None):
        if function_number==False:
           print(f"Accuracy diagrams for the normal_training whit one lost function")
           print(f"history={history}\nmodel={model_name}\nfunctions={function}\nfeatures={features_name}\nmodel_name={model_name}")
           common_library.plt.style.use('seaborn')
           common_library.plt.figure(figsize=(10,5))
           common_library.plt.plot(history.history['binary_accuracy'],label='train')
           common_library.plt.plot(history.history['val_binary_accuracy'],label='test')
           common_library.plt.plot([],[],' ',label='loss_function='+function)
           common_library.plt.plot([],[],' ',label='accuracy='+str(accuracy))
           common_library.plt.title(model_name+' whit '+features_name+' accuracy normal training whit one lost func')
           common_library.plt.ylabel('accuracy')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(loc='upper right')   
           common_library.plt.show()     
        if function_number==True:
           print(f"Accuracy diagrams for the normal_training whit multi lost function")
           print(f"history={history}\nmodel={model_name}\nfunctions={function}\nfeatures={features_name}\nmodel_name={model_name}")
           common_library.plt.style.use('seaborn')
           common_library.plt.figure(figsize=(10,5))
           k=0
           for i in history:
             common_library.plt.plot(i.history['binary_accuracy'],label='train '+function[k])
             common_library.plt.plot(i.history['val_binary_accuracy'],label='test '+function[k])
             k+=1
           k=0
           for i in function:
             common_library.plt.plot([],[],' ',label='acuracy '+i+'='+str(accuracy[i]))  
           common_library.plt.title(model_name+' whit '+features_name+' model accuracy normal training whit multy lost func ')
           common_library.plt.ylabel('accuracy')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(loc='upper right')   
           common_library.plt.show()  
        return 0
    def Loss_Diagrams(self,history,function,function_number=False,features_name=None,model_name=None):
        if function_number==False:
           print(f"Loss diagrams for the normal_training whit one lost function")
           print(f"history={history}\nmodel={model_name}\nfunctions={function}\nfeatures={features_name}\nmodel_name={model_name}")
           common_library.plt.style.use('seaborn')
           common_library.plt.figure(figsize=(10,5))
           common_library.plt.plot(history.history['loss'],label='train')
           common_library.plt.plot(history.history['val_loss'],label='test')
           common_library.plt.plot([],[],' ',label='loss_function='+function)
           common_library.plt.title(model_name+' whit '+features_name+' model loss normal training whit one lost func')
           common_library.plt.ylabel('loss')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(loc='upper right')   
           common_library.plt.show()          
        if function_number==True:
           print(f"Loss diagrams for the normal_training whit multi lost function")
           print(f"history={history}\nmodel={model_name}\nfunctions={function}\nfeatures={features_name}\nmodel_name={model_name}")
           common_library.plt.style.use('seaborn')
           common_library.plt.figure(figsize=(10,5))
           k=0
           for i in history:
             common_library.plt.plot(i.history['loss'],label='train '+function[k])
             common_library.plt.plot(i.history['val_loss'],label='test '+function[k])
             k+=1
           common_library.plt.title(model_name+' whit '+features_name+' model loss normal training whit multi lost func and')
           common_library.plt.ylabel('loss')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(loc='upper left')   
           common_library.plt.show() 
        return 0
    def Accuracy_Diagrams_Nkfold(self,history,function,function_number=False,features_name=None,model_name=None,cycles_nkfold=False,accuracy=None):
        if function_number==True and cycles_nkfold==False:   
           print(f"Accuracy diagrams for the nkfold training whit multi lost function and no cycles")
           print(f"Diagram for every fold")
           print(f"history={history}\nfunctions={function}\nfeatures={features_name}\nmodel={model_name}\ncycles={cycles_nkfold}")
           common_library.plt.style.use('seaborn')
           num_datasets = len(function)
           num_cols = int(common_library.np.ceil(common_library.np.sqrt(num_datasets)))
           num_rows = int(common_library.np.ceil(num_datasets / num_cols))
           common_library.plt.figure(figsize=(15, 5 * num_rows))
           k=1
           for i in history:
              p=0
              common_library.plt.subplot(num_rows, num_cols, k)
              for j in i:
                 label='K-fold='+str(p)
                 common_library.plt.plot(j.history["val_binary_accuracy"],label=label)
                 p+=1
              common_library.plt.plot([],[],' ',label='loss_function='+function[k-1])
              common_library.plt.plot([],[],' ',label='acuracy='+str(accuracy[function[k-1]]))
              common_library.plt.legend(loc='upper right')
              common_library.plt.ylabel('accuracy')
              common_library.plt.xlabel('epoch')
              common_library.plt.title(model_name+' whit '+features_name+' model accuracy '+str(len(history[0]))+'-fold whit one lost function')
              k+=1  
           common_library.plt.show() 
        if function_number==False and cycles_nkfold==False:
           print(f"Accuracy diagrams for the nkfold training whit one lost function and no cycles")
           print(f"Diagram for every fold")
           print(f"history={history}\nfunctions={function}\nfeatures={features_name}\nmodel={model_name}\ncycles={cycles_nkfold}")
           common_library.plt.style.use('seaborn')
           common_library.plt.figure(figsize=(10,5))
           k=0
           for i in history:
             label=("K-Fold="+str(k))
             common_library.plt.plot(i.history['val_binary_accuracy'],label=label)
             k+=1
           common_library.plt.plot([],[],' ',label='loss_function='+function)
           common_library.plt.plot([],[],' ',label='accuracy='+str(accuracy))
           common_library.plt.title(model_name+' whit '+features_name+' model '+'accuracy '+str(len(history))+'-fold whit one lost func')
           common_library.plt.ylabel('accuracy')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(loc='upper right')   
           common_library.plt.show() 

        if function_number==True and cycles_nkfold==True:
           print(f"Accuracy diagrams for the nkfold training whit multi lost function and cycles")
           print(f"history={history}\nfunctions={function}\nfeatures={features_name}\nmodel={model_name}\ncycles={cycles_nkfold}")
           num_datasets = len(function)
           num_cols = int(common_library.np.ceil(common_library.np.sqrt(num_datasets)))
           num_rows = int(common_library.np.ceil(num_datasets / num_cols))
           common_library.plt.figure(figsize=(15, 5 * num_rows))
           p=1
           for i in history:
              common_library.plt.subplot(num_rows, num_cols, p)
              k=0
              for j in i :
                 label='Cycle-'+str(k*2+1)+'acc-'+str(accuracy[k])
                 common_library.plt.plot(j['val_binary_accuracy'])
                 k+=1
              common_library.plt.plot([],[],' ',label='loss_function='+function[p])
              common_library.plt.plot([],[],' ',label='acc_mean-'+str(common_library.np.mean(accuracy[p:p+k],axis=0)))
              common_library.plt.title(model_name+' whit '+features_name+'acuracy for the 5-fold whit cycle')
              common_library.plt.ylabel('accuracy')
              common_library.plt.xlabel('epoch')
              common_library.plt.legend(loc='upper right') 
              p+=1  
           common_library.plt.show() 
        if function_number==False and cycles_nkfold==True:
           print(f"Accuracy diagrams for the nkfold training whit one lost function and cycles")
           print(f"history={history}\nfunctions={function}\nfeatures={features_name}\nmodel={model_name}\ncycles={cycles_nkfold}")
           common_library.plt.style.use('seaborn')
           common_library.plt.figure(figsize=(10,5))
           k=0
           for i in history:
              label='Cycle-'+str(k*2+1)+'acc-'+str(accuracy[k])
              common_library.plt.plot(i['val_binary_accuracy'],label=label)
              k+=1
           common_library.plt.plot([],[],' ',label='loss_function='+function)
           common_library.plt.plot([],[],' ',label='acc_mean-'+str(common_library.np.mean(accuracy,axis=0)))
           common_library.plt.title(model_name+' whit '+features_name+'acuracy for the 5-fold whit cycle')
           common_library.plt.ylabel('accuracy')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(loc='upper right')
           common_library.plt.show()
        return 0
    def Loss_Diagrams_Nkfold(self,history,function,function_number=False,features_name=None,model_name=None,cycles_nkfold=False):
        if function_number==True and cycles_nkfold==False:   
           print(f"Loss diagrams for the nkfold training whit multi lost function and no cycles")
           print(f"Diagram for every fold")
           print(f"history={history}\nfunctions={function}\nfeatures={features_name}\nmodel={model_name}\ncycles={cycles_nkfold}")
           common_library.plt.style.use('seaborn')
           num_datasets = len(function)
           num_cols = int(common_library.np.ceil(common_library.np.sqrt(num_datasets)))
           num_rows = int(common_library.np.ceil(num_datasets / num_cols))
           common_library.plt.figure(figsize=(15, 5 * num_rows))
           k=1
           for i in history:
              p=0
              common_library.plt.subplot(num_rows, num_cols, k)
              for j in i:
                 label='K-fold='+str(p)
                 common_library.plt.plot(j.history["val_loss"],label=label)
                 p+=1
              common_library.plt.plot([],[],' ',label='loss_function='+function[k-1])
              common_library.plt.legend(loc='upper right')
              common_library.plt.ylabel('loss')
              common_library.plt.xlabel('epoch')
              common_library.plt.title(model_name+' whit '+features_name+' model loss '+str(len(history[0]))+'-fold whit one lost function')
              k+=1  
           common_library.plt.show()         
        if function_number==False and cycles_nkfold==False:
           print(f"Loss diagrams for the nkfold training whit one lost function and no cycles")
           print(f"Diagram for every fold")
           print(f"history={history}\nfunctions={function}\nfeatures={features_name}\nmodel={model_name}\ncycles={cycles_nkfold}")
           common_library.plt.style.use('seaborn')
           common_library.plt.figure(figsize=(10,5))
           k=0
           for i in history:
             label=("K-Fold="+str(k))
             common_library.plt.plot(i.history['val_loss'],label=label)
             k+=1
           common_library.plt.plot([],[],' ',label='loss_function='+function)
           common_library.plt.title(model_name+' whit '+features_name+' model '+'accuracy '+str(len(history))+'-fold whit one lost func')
           common_library.plt.ylabel('loss')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(loc='upper right')   
           common_library.plt.show() 
        if function_number==True and cycles_nkfold==True:
           print(f"Loss diagrams for the nkfold training whit multi lost function and cycles")
           print(f"history={history}\nfunctions={function}\nfeatures={features_name}\nmodel={model_name}\ncycles={cycles_nkfold}")
           '''num_datasets = len(function)
           num_cols = int(common_library.np.ceil(common_library.np.sqrt(num_datasets)))
           num_rows = int(common_library.np.ceil(num_datasets / num_cols))
           common_library.plt.figure(figsize=(15, 5 * num_rows))
           for i in history:
              common_library.plt.subplot(num_rows, num_cols, k)
              for j in i :
                 for l in j:
                    common_library.plt.plot(l[''])
                 print()
           '''    
        if function_number==False and cycles_nkfold==True:
           print(f"Loss diagrams for the nkfold training whit one lost function and cycles")
           print(f"history={history}\nfunctions={function}\nfeatures={features_name}\nmodel={model_name}\ncycles={cycles_nkfold}")
           common_library.plt.style.use('seaborn')
           common_library.plt.figure(figsize=(10,5))
           k=0
           for i in history:
              label='Cycle-'+str(k*2+1)
              common_library.plt.plot(i['val_loss'],label=label)
              k+=1
           common_library.plt.plot([],[],' ',label='loss_function='+function)
           common_library.plt.title(model_name+' whit '+features_name+'loss for the 5-fold whit cycle')
           common_library.plt.ylabel('accuracy')
           common_library.plt.xlabel('epoch')
           common_library.plt.legend(loc='upper right')
           common_library.plt.show()
        return 0
          
'''
 def Accuracy_Model(self,history,function,Nkfold=False,function_number=False,features_name=None,model_name=None):
      print(function)
      path="./Diagrams_Accuracy_Loss/"
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
      path="./Diagrams_Accuracy_Loss/"
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
    def Accuracy_Model_Nkfold(self,history,function,function_number=False,features_name=None,model_name=None,cycles_nkfold=False):
           return 0
    def Loss_Digrams_Nkfold(self,history,function,function_number=False,features_name=None,model_name=None,cycles_nkfold=False):
           return 0
'''       

