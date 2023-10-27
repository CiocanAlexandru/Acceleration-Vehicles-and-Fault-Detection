# O clasa care contine functiile urmatoare
# functie de afisare instante in consola pentru fiecare sau  o categorie  
# functie care deseneaza o distributie a bazei de date  (graf cu culori pentru masini  cu defecte si nu pe categorii )
## Atentie !! Aici ar fi de preferat sa folosesti dictionarele din proiectul initial  ca sa fie mai usor de facut 
import common_library

class Data_Statistic:
    def __init__(self,data=None):
        self.data=data ## Data Manipulation object 
    def Number_of_Instance(self,info):
        if (self.data!=None):
            data=self.data.Get_Instances()
            k=0
            for i in range(0,len(data)):
                for j in data[i]:
                    if j==info:
                        k=k+1
            print("Instances whit number of:",k)
        else:
            print("Operation not working")
    def Dsitribution_Data_Base(self,info):
        if (self.data!=None):
            print(common_library.sns.displot(self.data.Get_DataFrame(),x=info))
        else:
            print("Operation not working")
    def Data_Frame_Diagram(self,data_frame,sample_rate):
        if (self.data!=None):
            print(common_library.librosa.waveplot(data_frame,sr=sample_rate))
        else:
            print("Operation not working")
        

