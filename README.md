# MNIST Classification Implementation in 3 Levels
Project Setup Details:
1) Install Python 3.10.xxx from the following link
   
   For Mac Os - https://www.python.org/downloads/macos/
   
   For Windows - https://www.python.org/downloads/windows/
2) once, you have the python installed and cloned the repository, navigate to the secure_federated_learning folder of the project and enter the following command

     pip install -r requirements.txt
3) If you are facing Python version issues because of the existing issues. Enter the following command for installing dependencies. The following command only works if you have an existing python 3.10.xx installed.

    python3.10 -m pip install -r requirements.txt

4) Note that if you have multiple Python versions, ensure your terminal points to the 3.10.xx version. If you are using Visual Studio code, you could easily change the interpreter by navigating to the view ==> command palette ==> select interpreter
5) If still Python issue persists, after making all the above changes, you could run python3.10 <filename.py>. Replace the <filename.py> with the file you would like to execute.

Code Structure Details:

core package contains the following:
1) Model for the Neural Network - NeuralNetwork.py
2) Federated Server - FederatedServer.py
3) Client Info - Client.py

Utilities Package contains the following:
1) General utility methods used accross all the implementations - utils.py
2) Utilities related to attack detection - attackutils.py

Files
1) For Level-1, the code is available at playground_level1.py
2) For Level-2, the code is available at playground_level2.py
3) For level-3, the code is available at playground_level3.py
4) Sample Attacks are available at attack.py

Code Execution Details:
1) Run playground_level1.py to view the results related to Level 1
2) Run playground_level2.py to view the results related to Level 2
3) Run playground_level3.py to view the results realted to Level 3
4) Run predict_image_level2.py to use the model and predict the digit. You can use the images from the images/ folder. Update the image you would like to use in the file and run it to see the predicted output
5) run attack.py to simulate an attack with a client of choice. Here detection methods are called for the parameters generated and the corresponding results are printed.

The model's state and client contents are stored in the following files. You can reuse them if you do not wish to train the model once again.
1) state corresponding to level 1 is stored with the name mnist_cnn.pth
2) state corresponding to level 2 is stored with the name federated_learning.pth
3) state corresponding to level 3 is stored with the name secure_federated_learning.pth
4) Client info related to level 2 is stored with the name clients_data.pth
5) Client info related to level 3 is stored with the name secure_clients_data.pth


Apart from this, you can also use the google colab notebooks for executing the same code. Here are the details for that.
1) Level 1: https://colab.research.google.com/drive/1Ud061WadMeYllSsK3VmmCitptVLhtPLx?usp=sharing
2) Level 2: https://colab.research.google.com/drive/1236C2EQQeXOHGca29Smik3PQUiFR49oY?usp=sharing
3) Level 3: https://colab.research.google.com/drive/198AjGZeMZUDVM5-U5XJO8tScZ0yLJspz?usp=sharing

Additional Notes:
1) If there are dependencies issue, please uncomment the first cell and upload the requirements.txt to runtime and execute it. That will solve the problem
2) If you wish to use the existing models, upload mnist_cnn.pth, federated_learning.pth, secure_federated_learning.pth so that model is not going to be trained again. 
3) For particular image prediction, create a folder called images and upload all the images and change the image name, that will do it.
