import header
from decimal import Decimal
import ast
data_set=None
FFT_exel=None
MFCC_exel=None
Extractor=None
PSD_exel=None
Diagrams_analisys=None

def Initialize():
    global data_set,FFT_exel,MFCC_exel,PSD_exel,Extractor,Diagrams_analisys
    print("Initialize variabels")
    data_set=header.Dataset_Function_Manipulation.Manipulation_Data_set("./exel/data_set.xlsx")
    FFT_exel=header.Dataset_Function_Manipulation.Manipulation_Data_set("./exel/FTT.xlsx")
    PSD_exel=header.Dataset_Function_Manipulation.Manipulation_Data_set("./exel/PST.xlsx") 
    MFCC_exel=header.Dataset_Function_Manipulation.Manipulation_Data_set("./exel/MFCC.xlsx")
    Extractor=header.Extract_Features_Augmentation.Features_Augmentation()
    Diagrams_analisys=header.Data_Statistics.Data_Statistic()
    return 0
def FFT_PSD_string_convertor(data):
   data_features=[]
   k=1
   for i in range(0,len(data)):
     print(f"Pas i={i}")
     converted_data=[[i] for i in data[i] if i!="['-']"]
     etichet=converted_data[len(converted_data)-1][0][1:len(converted_data[len(converted_data)-1][0])-1]
     instance=eval(etichet)
     all_data=""
     for j in range(0 ,len(converted_data)-1):
          for l in converted_data[j]:
             if l != '[' and l != ']' and l != "'" and l != '"' and l!=' ':
               all_data+=l
     converted_data=all_data[2:len(all_data)-2].split(",")
     vector=[]
     for j in range(0,len(converted_data)-1):
       try:
        vector.append(float(converted_data[j]))
       except ValueError:
          print("Exceptie")
          print(converted_data[j])
          k+=1
     vector=header.common_library.np.array(vector)
     data_features.append([vector,instance])
   print(data_features[1][0])
   print(len(data))
   print(f"Number of exceptions k={k}")
   data_features=header.common_library.np.array(data_features,dtype=object)
   return data_features
         
def MFCC_string_convertor(data):
   data_features=[]
   k=1
   for i in range(0,len(data)):
     print(f"Pas i={i}")
     converted_data=[[i] for i in data[i] if i!="['-']"]
     etichet=converted_data[len(converted_data)-1][0][1:len(converted_data[len(converted_data)-1][0])-1]
     instance=eval(etichet)
     print(instance)
     all_data=""
     for j in range(0 ,len(converted_data)-1):
          for l in converted_data[j]:
             if l != '[' and l != ']' and l != "'" and l != '"' and l!=' ':
               all_data+=l
     converted_data=all_data[2:len(all_data)-2].split("|")
     matrix=[]
     for j in range (0,len(converted_data)):
          one_line=converted_data[j].split(",")
          new_line=[]
          for l in one_line[0:len(one_line)-1]:
              try:
               new_line.append(float(l))
              except ValueError:
                 k+=1
                 print("Exception")
          new_line=header.common_library.np.array(new_line)
          matrix.append(new_line)
     matrix=header.common_library.np.array(matrix)  
     data_features.append([matrix,instance])
   print(len(data_features))
   print(f"Number of exeception k={k}")
   data_features=header.common_library.np.array(data_features,dtype=object)
   return data_features

Initialize()

FFT_ok=input("Use FFT features?(Yes/No)")
MFFC_ok=input("Use MFFC features?(Yes/No)")
PSD_ok=input("USE PSD features?(Yes/No)")


if FFT_ok.lower()=="yes":
    data1=FFT_exel.Get_Instances()
    data=FFT_PSD_string_convertor(data1)
elif MFFC_ok.lower()=="yes":
    data1=MFCC_exel.Get_Instances()
    data=MFCC_string_convertor(data1)
elif PSD_ok.lower()=="yes":
     data1=PSD_exel.Get_Instances()
     data=FFT_PSD_string_convertor(data1)
Model_CNN_ok=input("Use CNN model?(Yes/No)")
Model_SVN_ok=input("Use SVM model?(Yes/No)")
Model_FCNN_ok=input("USE FCNN model?(Yes/No)")

if Model_CNN_ok.lower()=="yes":
   print("Model_CNN")
elif Model_SVN_ok.lower()=="yes":
   print("Model_SVN")
elif Model_FCNN_ok.lower()=="yes":
   print("Model_FCNN")

data,class_encoder=Extractor.hot_encoding(data)
print(len(class_encoder))
print(class_encoder)








