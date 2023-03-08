# FER
This is the repository for the project "Artificial Intelligence for Affective Computing".

## Datasets
The model is trained on [RAF-DB](http://www.whdeng.cn/raf/model1.html). 

[comment]: <> (The self-constructed [multifaceted dataset]&#40;http://www.whdeng.cn/raf/model1.html&#41; can be downloaded. )

## Usage
### Preprocess
To preprocess the RAFDB dataset:
1. Go to /dataset/original.
2. Download the aligned RAFDB dataset and add it to a new folder called RAFDBAligned.
3. Put the downloaded list_patition_label.txt file in the folder /dataset/original.
4. Run preprocess/split.py to split the dataset into training, validation and test sets. The outputs are stored in /dataset/RAFDBAligned.

To preprocess and extract the detected faces of the RAFDB dataset:
1. Go to /dataset/original.
2. Download the RAFDB dataset and add it to a new folder called RAFDB.
3. Put the downloaded list_patition_label.txt file in the folder /dataset/original.
4. Run preprocess/split_and_refined.py to split the dataset into training, validation and test sets. The outputs are stored in /dataset/RAFDBRefined.

### Train
Run train.py in /train. The saved models, json file recording the training process, corresponding graphs are stored in /checkpoints.

### Test
To test the saved model on the test set:
Run test.py in /test. The corresponding confusion matrix is stored in /output.

### Attention Map Visualization
To visualize the attention map of the saved model:
Run visualize.py in /utils. The attention map is stored in /output.
