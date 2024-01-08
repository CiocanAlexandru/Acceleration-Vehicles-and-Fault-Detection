<?php
require '../models/models_Prediction.php';
$mod=1;
$checking=true;
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
 
 if ($checking==false)
  echo "Not set";
else 
  GetPrediction();
 
 /*$_POST["model"]=null;
 $_POST["train"]=null;
 $_POST["fueatures"]=null;
 $_POST["lost-function"]=null;
 $_POST["number_of_cycel"]=null;
*/
require '../view/Prediction.php' ;
?>