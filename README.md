# CLIPyourFood
Used
[GitHub](https://youtu.be/i8Cnas7QrMc)

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
```bash
pip install -r requirements.txt
```
### 2. Download the dataset
The used dataset is FOOD101.<br />
Run the Data/utils.py file as main for downloading the dataset.

### 3. Run
#### Train model
You can modify hyperparameters in the relevant block at train_model.py. <br />
Run the train_model script for training model. <br />
You can add to the model CLIP features of image. <br />
If you have dish name info it can be also injected to the model with text encoder of CLIP.<br />
#### Test model
Modify the parameters as model path and set the configuration 
for testing the model on test set of FOOD101.<br />
The scores are printed to the output channel (default terminal).
