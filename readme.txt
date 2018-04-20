<><><><><><><><><><><><><><><><><><><><><><><><><><><><>
|        Saman Sadeghi Afgeh & Shakeel Raja			   |
<><><><><><><><><><><><><><><><><><><><><><><><><><><><>

Folder content: 

The experiment  is controlled by the main.m file. This file uses the functions SupportVectorMachine and MultilayerPerceptron, which are stored in the files MultilayerPerceptron.m and SupportVectorMachine.m. 

Data normalization has been performed using StatisticalNormaliz.m function file  located in the folder func. 

Initial data cleaning and some pre-processing steps, along with visualization has been performed in python with BC_clean.py file which uses BreastCancerData.csv and outputs BreastCancerData_Clean.csv which is then used by our main algorithms.  Original and clean datasets are also included. 

To run the program, open main.m, select current working directory in MATLAB, and run the code. All other functions will be automatically imported from the respective files.  

--------------------------------------------------------
Note: 
The function crossvalind(), used for MLP Cross-validation code in MultilayerPerceptron.m  , requires the Bionformatics Toolbox to be installed. 

The the inspiration for cross validation has been taken from Gregg Heath's solution from mathworks groups. the original idea with code can be viewed at:
http://uk.mathworks.com/matlabcentral/newsreader/view_thread/340857

CAUTION : The Grid search sections in main.m may take some time to run as these run the classifiers thousands of times to find the optimal model 
