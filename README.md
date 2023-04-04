# FER
This is the repository for the project "Artificial Intelligence for Affective Computing".

## Datasets
Pre-training is performed on [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
The model is trained and tested on [RAF-DB](http://www.whdeng.cn/raf/model1.html). 

The multi-faced expression recognition dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/10uAtja-Qo-qb6NLan69C5QjOV7EiiSqZ/view?usp=sharing).

## Training
The pretrained weights are from swinv2_small_window16_256.

### Preprocessing
To split the aligned RAFDB dataset:
1. Go to /dataset/RAFDB/Image.
2. Download the aligned RAFDB dataset and add it to a new folder called aligned.
3. Put the downloaded list_patition_label.txt file in the folder /dataset/RAFDB.
4. Run preprocess/split.py to split the dataset into training, validation and test sets. The outputs are stored in /dataset/RAFDBAligned.

To split the original RAFDB dataset and align the detected faces:
1. Go to /dataset/RAFDB/Image.
2. Download the original RAFDB dataset and add it to a new folder called original.
3. Put the downloaded list_patition_label.txt file in the folder /dataset/RAFDB.
4. Run preprocess/split_and_refined.py to split the dataset into training, validation and test sets. The outputs are stored in /dataset/RAFDBRefined.

### Train the system with different configurations
Run train.py in /script. The saved models, json file recording the training process, corresponding graphs are stored in /checkpoints.\
The default configuration is the original small SwinV2 model.

To add the CBAM module into SwinV2, change the model_name to 'cbam'.\
To add the SE module into SwinV2, change the model_name to 'se'.

### Train the system with different loss functions
Run train_contrastive_learning.py in /script. The saved models, json file recording the training process, corresponding graphs are stored in /checkpoints.
The default loss function is center loss.

### Train the system with knowledge distillation loss
1. Run train.py in /script with model_name='distill_base'.
2. Load the saved teacher model and run train_distillation.py in /script.

## Testing
### Test the saved model on the test set of RAFDB
Run test.py in /script. The corresponding confusion matrix is stored in /output.

### Test the saved model on the multi-faced dataset
Run test_multiface.py in /script. The accuracy and match ratio are printed in terminal.

## Evaluation
### Attention Map Visualization
To visualize the attention map of the saved model:
1. Clone the repository [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam.git).
2. Put the file visualize.py in /utils into the folder pytorch-grad-cam and run it. The attention map is stored in /output.

### User Interface
To test the saved model with a simple user interface:
Run interface.py