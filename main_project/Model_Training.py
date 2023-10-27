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
    #MFCC_exel=header.Dataset_Function_Manipulation.Manipulation_Data_set("./exel/MFCC.xlsx")
    Extractor=header.Extract_Features_Augmentation.Features_Augmentation()
    Diagrams_analisys=header.Data_Statistics.Data_Statistic()
    return 0
def FFT_PSD_string_convertor(data):
   data_features=[]
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
     converted_data=all_data[2:len(all_data)-3].split(",")
     vector=[]
     for j in range(0,len(converted_data)-1):
       try:
        vector.append(float(converted_data[j]))
       except:
          print("Exceptie")
     data_features.append([vector,instance])
   print(data_features[1][0])
   print(len(data))
   return data_features


def All_is_float(data):
    for i in data:
       print(i[1])
       for j in i[0]:
          if type(j) is not float:
             return False
    return True          
def MFCC_string_convertor(data):
  print(data)


Initialize()

FFT_ok=input("Use FFT features?(Yes/No)")
MFFC_ok=input("Use MFFC features?(Yes/No)")
PSD_ok=input("USE PSD features?(Yes/No)")


if FFT_ok.lower()=="yes":
    data1=FFT_exel.Get_Instances()
    data=FFT_PSD_string_convertor(data1)
    print(f"Are all element float ? {All_is_float(data)}")
elif MFFC_ok.lower()=="yes":
    data1=MFCC_exel.Get_Instances()
    data=MFCC_string_convertor(data1)
elif PSD_ok.lower()=="yes":
     data1=PSD_exel.Get_Instances()
     data=FFT_PSD_string_convertor(data1)
     print(f"Are all element float ? {All_is_float(data)}")
Model_CNN_ok=input("Use CNN model?(Yes/No)")
Model_SVN_ok=input("Use SVM model?(Yes/No)")
Model_FCNN_ok=input("USE FCNN model?(Yes/No)")

if Model_CNN_ok.lower()=="yes":
   print("Model_CNN")
elif Model_SVN_ok.lower()=="yes":
   print("Model_SVN")
elif Model_FCNN_ok.lower()=="yes":
   print("Model_FCNN")








