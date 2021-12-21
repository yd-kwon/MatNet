
# MatNet

This repository provides a reference implementation of *MatNet* and saved trained models as described in the paper:<br>
> Matrix Encoding Networks for Neural Combinatorial Optimization <br>
(NeurIPS 2021, accepted)<br>
https://arxiv.org/abs/2106.11113


The code is written using Pytorch.<br>
<br>

## Getting Started
We provide codes for two CO (combinatorial optimization) problems:

* Asymmetric Traveling Salesman Problem (ATSP)
* Flexible Flow Shop Problem (FFSP) 

### Basic Usage
For both ATSP_MatNet and FFSP_MatNet, <br>

   #### i. To train a model, 
   ```
   python3 train.py
   ```
   train.py contains parameters you can modify. <br>
   At the moment, it is set to train N=20 problems.

   
   #### ii. To test a model,
   ```
   python3  test.py
   ```
   You can specify the model as a parameter contained in test.py. <br>
   At the moment, it is set to use the saved model (N=20) we have provided (in "result" folder), but you can easily use the one you have trained from running train.py.

   To test for the N=50 model, make sure that saved problem files exist in the path (see below for Datasets). Also, modify test.py so that 
   * all "20"'s are changed to "50"
   * "path" and "epoch" in the "tester_params" are correctly pointing to the saved model
   * test batch size is decreased (by a factor of something like 4)



### Saved Trained Models
   Trained models for ATSP are provided with the codes.<br>
   However, the sizes of the trained model files for FFSP-MatNet are too large to upload here. So we provide them as links below. 
   * [MatNet_saved_models_FFSP.zip](https://drive.google.com/file/d/1tvAtDG8MEfwyuSjAVWjw6ZoEGOAHTZSy/view?usp=sharing) (size: 358MB) <-- a zip file containing all three models
      * [MatNet_saved_models_N100](https://drive.google.com/drive/folders/1Iymra5DjVfZe_lAXO7NcZnSa-foZjqBP?usp=sharing) (size: 119MB)
      * [MatNet_saved_models_N50](https://drive.google.com/drive/folders/1X8IrBN4KPmbd9vkizv-kGVx_PiRIAZ_M?usp=sharing) (size: 119MB)
      * [MatNet_saved_models_N20](https://drive.google.com/drive/folders/1091JQMZmYjYn1BiqDYqDt6u_qrVTtY9F?usp=sharing) (size: 119MB)


### Datatsets
Test datasets for larger N (N=50, N=100) problems are given as links below.<br>
  Download the dataset and add them to the "data" directories under ATSP and FFSP folders. <br>
  * [MatNet_dataset_ATSP](https://drive.google.com/file/d/1NLrck1NU3rQ9_oraK0eKgT4DcLCvWeU4/view?usp=sharing) (size: 374MB)
  * [MatNet_dataset_FFSP](https://drive.google.com/file/d/1MBf1fgquDLwUvS-75h36C3Rir7SFGutj/view?usp=sharing) (size: 19.6MB)
  

### Used Libraries
python v3.7.6 <br>
torch==1.7.0 <br>

