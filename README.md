# CLIPyourFood
Used repository in this project OpenAI-CLIP
[GitHub](https://github.com/openai/CLIP) <br />
Helping repositories along the way: [Classifier-GitHub](https://github.com/mandeer/Classifier), [food-101-GitHub](https://github.com/shubhajitml/food-101/blob/master/food-101-pytorch.ipynb) <br />


## Prerequisites
| Library                | Version |
|------------------------|---------|
| `Python`               | `3.9`   |
| `cuda (for GPU usage)` | `11.3 ` |

## installation guide
### 1. Virtual Environment
#### 1.1. Create a virtual environment
```bash
python3 -m venv venv
```
#### 1.2. Activate the virtual environment
```bash
source venv/bin/activate
```
#### 1.3. Install the requirements
Install requirements and GUI for display result on sample.
```bash
pip install -r requirements.txt
sudo apt-get install python3-tk
```
### 2. Download the dataset
The used dataset is [FOOD101](https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html#:~:text=The%20Food%2D101%20is%20a,contain%20some%20amount%20of%20noise.).<br />
Run the Data/utils.py file as main for downloading the dataset.
* You can clean the data [annotations](http://www.ub.edu/cvub/ingredients101/) by yourself with Data/annotation_extractor.py.
##  Train & Test
#### Train model
Mdify hyperparameters in the relevant block at `train_model.py`. <br />
The output path should exist. 
Run the train_model script for training model. <br />
You can add to the model CLIP features of image. <br />
If you have dish name info it can be also injected to the model with text encoder of CLIP.<br />
#### Test model on test set
Modify the parameters at `test_model.py` as model path and set the configuration 
for testing the model on test set of FOOD101.<br />
The scores are printed to the output channel (default terminal).
#### Test model on single sample
Modify the parameters section in `sample_test.py`. <br />
The parameters relate to the loaded model and the image path. <br />
The result will be displayed on separate window with the ingredients as the title.
## Files in the repository

| File name                                                     | Purpsoe                                                           |
|---------------------------------------------------------------|-------------------------------------------------------------------|
| `train_model.py`                                              | train Resnet18 model with configuration from CLIP                 |
| `test_model.py`                                               | main application tailored for Atari's Pong                        |
| `sample_test.py`                                              | main application tailored for Atari's Boxing                      |
| `Data/IngredientsLoader.py`                                   | sample code for playing a game, also in `ls_dqn_main.py`          |
| `Data/utils.py`                                               | classes for actions selection (argmax, epsilon greedy)            |
| `Data/annotation_extractor.py`                                | agent class, holds the network, action selector and current state |
| `Data/Ingredients_json/...`                                   | DQN classes, neural networks structures                           |
| `model/BasicNodule.py`                                        | Replay Buffer classes                                             |
| `model/Resnet.py`                                             | hyperparameters for several Atari games, used as a baseline       |
| `model/Resnet_w_concat_connection.py`                         | Shallow RL algorithms, LS-UPDATE                                  |
| `model/utils.py`                                              | utility functions                                                 |