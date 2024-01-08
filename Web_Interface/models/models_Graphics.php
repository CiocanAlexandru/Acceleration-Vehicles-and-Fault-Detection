<?php

function GetPath($model,$features,$train,$lost_function,$cycles)
{
   $content="";
   $height=400;
   $whight=525;
   if ($model!="SVM")
   { 
     if ($train==="Normal" && $lost_function==="One function")
      {
        $content="../../Diagrams_Accuracy_Loss/Accuracy_".$model."_".$features."_"."normaltrainng_one_lost_function.jpg";
      }
     if($train==="Normal" && $lost_function!="One function")
       {
         $content="../../Diagrams_Accuracy_Loss/Accuracy_".$model."_".$features."_"."normaltrainng_multi_lost_function.jpg";
         $whight=750;
       }

     if ($train==="Nkfold" && $lost_function==="One function" && $cycles==="One cycle")
       {
        $content="../../Diagrams_Accuracy_Loss/Accuracy_".$model."_".$features."_"."kfold_one_lost_function_nocycles.jpg";
       }
     if ($train==="Nkfold" && $lost_function==="One function" && $cycles!="One cycle")
       {
        $content="../../Diagrams_Accuracy_Loss/Accuracy_".$model."_".$features."_"."kfold_one_lost_function_yescycles.jpg";
       }
     if ($train==="Nkfold" && $lost_function!="One function" && $cycles==="One cycle")
      {
       $content="../../Diagrams_Accuracy_Loss/Accuracy_".$model."_".$features."_"."kfold_multi_lost_function_nocycles.jpg";
       $whight=750;
      }
     if ($train==="Nkfold" && $lost_function!="One function" && $cycles!="One cycle")
      {
       $content="../../Diagrams_Accuracy_Loss/Accuracy_".$model."_".$features."_"."kfold_multi_lost_function_yescycles.jpg";
       $whight=750;
      }
   }
   if($model==="SVM")
   {

    if ($train==="Normal")
      {
        $content="../../Diagrams_Accuracy/Normal_Training_Average_Confusion_Matrix_SVM_whit_".$features.".jpg";
      }
    if ($train==="Nkfold" && $cycles==="One cycle")
      {
        $content="../../Diagrams_Accuracy/SVM_whit_kfold_no_cycles_".$features.".jpg";
      }
    if ($train==="Nkfold" && $cycles!="One cycle")
      {
       $content="../../Diagrams_Accuracy/SVM_whit_kfold_yes_cycles_".$features.".jpg";
      }

   }
   $result=[];
   array_push($result,$content);
   array_push($result,$height);
   array_push($result,$whight);
   return $result;
}

?>