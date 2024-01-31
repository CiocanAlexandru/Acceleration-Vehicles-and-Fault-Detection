<?php
require '../models/models_Prediction.php';
$mod=1;
$checking=true;
if (isset($_POST["model"])==false )
 {
    $checking=false;
 }
 

 if (isset($_POST["features"])==false )
 {
    $checking=false;
 }
 

 if (isset($_POST["train"])==false)
 {
    $checking=false;
 }
 

 if (isset($_POST["lost-function"])==false )
 {
    
    $checking=false;
 }
 

 if (isset($_POST["number_of_cycel"])==false )
 {
    $checking=false;
 }
 //if(isset($_FILES["fileToUpload"])!=true || $_FILES["fileToUpload"]["error"] != 0) 
 //{
 //  $checking=false;
 //}
if ($checking==true) 
{
   $model=GetPrediction($_POST["model"],$_POST["features"],$_POST["train"],$_POST["lost-function"],$_POST["number_of_cycel"]);
   $file_path=SedUploadFile($_FILES["fileToUpload"]);
   //function Prediction($features,$model_name,$audiofile)
   $content=Prediction($_POST["features"],$model,$file_path,$_POST["model"]);

}
 
 /*$_POST["model"]=null;
 $_POST["train"]=null;
 $_POST["fueatures"]=null;
 $_POST["lost-function"]=null;
 $_POST["number_of_cycel"]=null;
*/
require '../view/Prediction.php' ;
?>