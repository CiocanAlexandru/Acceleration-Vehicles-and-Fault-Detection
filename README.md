<h1># Acceleration-Vehicles-and-Fault-Detection</h1>
<body>
<a href="https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646848.pdf">Documentatie</a>
<hr>
<a href="https://cs229.stanford.edu/proj2019aut/data/assignment_308875_raw/26504237.pdf">Poster</a>
<hr>
<a href="https://github.com/CiocanAlexandru/Licenta">Old project</a>

<h1>Introducere</h1>
<p>

Scopul acestui proiect este  să exploreze utilizarea analizei semnalelor audio pentru clasificarea mașinilor și stabilirea stării tehnice, tipului și brandului, într-un mod non-invaziv și eficient. Scopul principal este de a identifica natura și complexitatea problemei abordate și de a justifica alegerea unei abordări multi-clasă.
 Astfel acest repo contine implementarea modelelor,generearea de noi isntante de antrenare ,baza de date in  sine (inregistrarile audio) si implementarea pentru interfata web.Pentru cei care le place partea mai tehnica si matematica si sunt dorniti sa cunoasca mai bine lucrurile este atasat mai jos linkiri catre documentatea si o doagrama UML a interactiuni componentelor implemnetarea find bazata pe paradigma programarii orientate pe obiect srespectand pe cat posibil princiliile SOLID .

</p>
<h1>Structura folder</h1>
<p>Din cate se poate observa proiectul contine mai multe foldere care detin informati de interas ca o scurta prezentare ce contine fiecare avem urmatoarele :</p>
<ul>
<li>DiagramsWav= contine diagrame cu reprezentarea inregistrailor ,diagrame de distributie,digrame cu transformari</li>
<li>Diagrams_Accuracy_Loss=contine diagrame cu evolutie a antrenari modelelor cu diferite metode de caracteristici si diferite metode de antrenare</li>
<li>Models=modele salvate dupa antrenare si refolosite pentru interfata web</li>
<li>Web_Interface=implemnetarea pentru interfata web</li>
<li>exel=exeluri separate pentru cele trei metode de extragere a caracteriticilor</li>
<li>main_project=implemnetarea propriuzisa  proiectului </li>
<li>wav_=cintine inregistraile audio cu etichete [tip,brand,state]</li>
</ul>
<h1>Scenari de utilizare</h1>
Pentru ca componenta web sa functioneze cum trebuie este indicat sa fie folositi pasi de mai jos  asta daca se adauga inregistrai noi la baza de date 
<h2>Mediu de lucru</h2>
Ca sa poata fi utilizata aplicatia trebuie sa fie instalate urmatoarele :
<ul>
<li>Python veriune 3.10.11 sau mai noua </li>
<li>Un IDE de exemplu am folosit Visual Studio Code</li>
<li>XAMPP un tool pentru server local</li>
<li>Librari pentru modele si date</li>
</ul>
<p>Note!!
Pentru  a fii mai usor instalarea bibliotecilor  pentru python aveti comenzile de instalare mai jos care trebuie date in terminalul proiectului unde este codul:</br>
pip install numpy</br>
pip install keras</br>
pip install librosa</br>
pip install pandas</br>
pip install seaborn</br>
pip install tensorflow</br>
pip install matplotlib</br>
pip install scikit-learn</br>
pip install joblib</br>
pip install scipy</br>
pip install libsvm</br>
pip install openpyxl
</p>
<h2>Pas1- Generarea de date noi, antrenare,si rulare</h2>

 <h3>Pas 1.1</h3> <p>Daca se doreste adaugarea de inregistrai noi se dauga in dorectorul specific indicat mai sus inregitrarea sau inregistraile trebuie dupa sa fie generate
 exelurile specifice cu ajutorul scriputul <strong>Data_Load_Augmentation.py</strong> unde in consola se dau optiuni si se poate genera in functie de preferinte care metoda de extragere se doreste
 </p>
 <h3>Pas 1.2</h3>
 <p>Dupa ce sau generat noile exeluri sau se vrea antrenarea si salvarea noilor modele si diagrame se va face cu ajutorul scriptului <strong>Model_Training.py</strong> care tot in consola vor fi date optiuni de la metode de extragere 
 pana la ce fel de antrenare se vrea normala nkfold cu una sau mai multe functi de pierdere sau un singur ciclu sau mia multe la nkfold 
 </p>
 <h3>Pas 1.3(Optional!)</h3> 
 <p>Daca se vrea o vizualizare a bazei de date sau generare si reprezentare a unei transformari sau inregistrai se va folosi scriptul  <strong>Statistic_Wave_Diagram.py</strong> tot asa cu optiuni specifice </p>

<h2>Pas2-Utilizarea Interfetei Web</h2>
</body>
![Change location dorectory XAMPP](https://github.com/CiocanAlexandru/Acceleration-Vehicles-and-Fault-Detection/assets/79212626/d77724bc-a2fd-490a-aebc-3f88ee914b9d)
![Deschidere XAMPP](https://github.com/CiocanAlexandru/Acceleration-Vehicles-and-Fault-Detection/assets/79212626/c16e6032-c810-4f56-9406-3e33a8c79dc1)
![httoc XAMPP open](https://github.com/CiocanAlexandru/Acceleration-Vehicles-and-Fault-Detection/assets/79212626/be035bff-db1d-48df-afd5-b8e6b7767eb8)
![Start XAMPP](https://github.com/CiocanAlexandru/Acceleration-Vehicles-and-Fault-Detection/assets/79212626/a9605d9b-f67b-45c4-b60a-4248a7731a12)


