# FER
This is the repository for the project "Artificial Intelligence for Affective Computing".

## Datasets
The model is trained on [RAF-DB](http://www.whdeng.cn/raf/model1.html) and pre-trained on [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). 
The multifaceted expression recognition dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/10uAtja-Qo-qb6NLan69C5QjOV7EiiSqZ/view?usp=sharing).

## Usage
### Preprocess
To preprocess the RAFDB dataset:
1. Go to /dataset/RAFDB/Image.
2. Download the aligned RAFDB dataset and add it to a new folder called aligned.
3. Put the downloaded list_patition_label.txt file in the folder /dataset/RAFDB.
4. Run preprocess/split.py to split the dataset into training, validation and test sets. The outputs are stored in /dataset/RAFDBAligned.

To preprocess and extract the detected faces of the RAFDB dataset:
1. Go to /dataset/RAFDB/Image.
2. Download the RAFDB dataset and add it to a new folder called original.
3. Put the downloaded list_patition_label.txt file in the folder /dataset/RAFDB.
4. Run preprocess/split_and_refined.py to split the dataset into training, validation and test sets. The outputs are stored in /dataset/RAFDBRefined.

### Train
Run train.py in /script. The saved models, json file recording the training process, corresponding graphs are stored in /checkpoints.

### Test
To test the saved model on the test set:
Run test.py in /script. The corresponding confusion matrix is stored in /output.

### Attention Map Visualization
To visualize the attention map of the saved model:
Run visualize.py in /utils. The attention map is stored in /output.
