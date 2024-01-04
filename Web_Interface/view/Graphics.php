<!DOCTYPE html>
<html lang="en">
    <head>
    <title>
         Acceleration Vehicles
    </title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <link rel="icon" type="image/x-icon" href="../images/favicon.png">
    <link rel="stylesheet" href="../styles/style.css">
    <link rel="stylesheet" href="../styles/Graphics.css">
    </head>
    <body>
     <div class="upper">
        <img src="../images/Above Image.jpg">
        <p><a href="Home">Acceleration Vehicles and Fault Detection</a></p>
    </div>
    <div>
        <ul>
            <li><a href="Home">Home</a></li>
            <li><a href="Prediction">Prediction</a></li>
            <li><a href="Audio">Audio</a></li>
            <li><a href="Graphics">Graphics</a></li>
            <li><a href="About">About us</a></li>
        </ul>
    </div>
    <section>
        <div class="content">
        <h1>Graphics</h1>
            <p><strong>Let's see some graphics performens</strong></p>
            <div class="graphics" accept=".wav">
            <img src="../images/Fundal Image.jpeg">
            </div>
            <form action="" method="Post">
            <div class="options">
            <label for="model">Model</label>
            <select id="model" name="model">
            <option value="FCNN">FCNN</option>
            <option value="CNN">CNN</option>
            <option value="SVM">SVM</option>
            </select>
            
            <label for="Fueatures">Fueatures</label>
            <select id="Fueatures" name="Features">
            <option value="FFT">FFT</option>
            <option value="MFCC">MFCC</option>
            <option value="PSD">PSD</option>
            </select>

            <label for="Train">Traing Method</label>
            <select id="Train" name="Train">
            <option value="Normal">Normal</option>
            <option value="Nkfold">Nkfold</option>
            </select>

            <label for="Lost-Function">Lost-Function</label>
            <select id="Lost-Function" name="Lost-Function">
            <option value="One-Function">One function</option>
            <option value="Multi-Function">Multi Function</option>
            </select>
            <label for="Number of cycel">Cycel nkfold</label>
            <select id="Number of cycel" name="Number of cycel">
            <option value="One-Function">One cycel</option>
            <option value="Multi-Function">More cycles 3 5 7 11</option>
            </select>
            <button type="submit" class="submit-button">Submit</button>
            </div>
            </form>
           
        </div>
    </section>
    <footer>
        <p>I made you interested <br> more info  <a href="About">here!</a></p>
    </footer>
    </body>
</html>