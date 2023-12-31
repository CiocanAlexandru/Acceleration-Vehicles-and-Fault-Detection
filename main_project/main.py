import header 

path_file=None
model=None
feature_method=None
number_loss_function=None
type_of_training=None
number_of_cycles=None
site_mod_type=None
audio=None
Extractor=None
def Initialize():
  global model,feature_method,number_loss_function,type_of_training,number_of_cycles,site_mod_type,Extractor,audio
  model=header.common_library.sys.argv[1]
  feature_method=header.common_library.sys.argv[2]
  number_loss_function=header.common_library.sys.argv[3]
  type_of_training=header.common_library.sys.argv[4]
  number_of_cycles=header.common_library.sys.argv[5]
  site_mod_type=header.common_library.sys.argv[6]
  audio=header.common_library.sys.argv[7]
  Extractor=header.Extract_Features_Augmentation.Features_Augmentation()
##Where i predict using the models saves 
def Get_Result(audio,path_file,feature_method):
   result=None
   class_index_only=Extractor.index_class_from_file()
   sample_rate,audio_data=Extractor.read_data(audio)
   if feature_method.lower()=='ftt':
      data=Extractor.FFT_Futures(audio_data,sample_rate)
   if feature_method.lower()=='mfcc':
      data=Extractor.MFFC_Features(audio_data,sample_rate)
   if feature_method.lower()=='psd':
      data=Extractor.PSD_Features(audio_data,sample_rate)
   return result

Initialize()
if site_mod_type.lower=="Predict":
   path_file="./Models/model_"
result=None
if model:
   path_file+=model

if feature_method:
   path_file+=feature_method

if number_loss_function:
   path_file+=number_loss_function

if type_of_training:
   path_file+=type_of_training

if number_of_cycles:
   ## model whit multiple loss functions
   loss_functions=[]
   path_file+=number_of_cycles

print(Get_Result(audio,path_file,feature_method))
