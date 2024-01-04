<?php
/*
function CheckURL($Features,$Project_Name,$arr)
{
    $poz1=0;$poz2=0;$coun=0;$k=0;
    foreach($arr as $i)
    {
        if ($i===$Project_Name)
        {
            $count=$count+1;
            $poz1=$k;
        }
        if ($i===$Features)
        {
            $count=$count-1;
            $poz2=$k;
        }  
        if ($i!=$Features && $i!=$Project_Name && $i!="localhost")
          echo "Difreant";
        $k=$k+1;  
    }
    if ($count!=0)
      return false;
    if ($poz1>$poz2)
      return false;
    return true;
};
*/
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
