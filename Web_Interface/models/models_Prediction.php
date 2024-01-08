<?php

function GetPrediction()
{
    exec("python ../../main_project/main.py  value",$output,$ok_status);
    if ($ok_status===0)
    {
        foreach ($output as $i)
        {
            echo $i;
        }
    }
    else {
        echo "Execution error";
    }

    return "";
}
$ceva=1;
echo "";

?>