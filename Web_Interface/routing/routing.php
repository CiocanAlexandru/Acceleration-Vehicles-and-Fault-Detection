<?php

$arr= explode("/", $_SERVER['REQUEST_URI'] );

if (in_array("Home", $arr))
    require '../controllers/controller_index.php';
elseif (in_array("About", $arr))
    require "../controllers/controller_About_us.php";
elseif (in_array("Audio", $arr))
    require "../controllers/controller_Audio.php";
elseif (in_array("Prediction", $arr))
    require "../controllers/controller_Prediction.php";
elseif (in_array("Graphics", $arr))
    require "../controllers/controller_Graphics.php";
else 
 echo "Wrong Adrres";

?>