# PGCNMDA
Predicting miRNA-disease associations based on graph convolutional network with path learning

## Requirements
  * python==3.9.15
  * dgl==0.9.1
  * networkx==2.8.8
  * numpy==1.23.5
  * scikit-learn==1.1.3
  * pytorch==1.12.0
  * tqdm==4.64.1
  * pandas==1.5.2
  * openpyxl==3.1.2

## File
### data
  We provide the HMDD v2.0 dataset, which contains 495 miRNAs and 383 diseases, with 5430 associations between them. 
  * known disease-miRNA association number.txt:Validated mirNA-disease associations  
  * disease semantic similarity matrix 1.txt: The first kind of disease semantic similarity
  * disease semantic similarity matrix 2.txt: The second kind of disease semantic similarity
  * disease functional similarity matrix.txt: Disease functional similarity
  * miRNA functional similarity matrix.txt: MiRNA functional similarity	
  * disease number.txt: Disease id and name
  * miRNA number.txt: MiRNA id and name

### model
  We provide the code to construct PGCN model.
  * PGCN.py: Construct SFGAE model

### 5-CV
  We provide the code to implement 5-CV experximents.
  * main.py: The startup code of the program
  * train.py: Train the model
  * utils.py: Methods of data processing

### 10-CV
  We provide the code to implement 10-CV experximents.
  * main.py: The startup code of the program
  * train.py: Train the model
  * utils.py: Methods of data processing
  
### GLOOCV
  We provide the code to implement GLOOCV experximent.
  * main.py: The startup code of the program
  * train.py: Train the model
  * utils.py: Methods of data processing

## Contact
If you have any questions or comments, please feel free to email Cheng Yan(yancheng01@hnucm.edu.cn).