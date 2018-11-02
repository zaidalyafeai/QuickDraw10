# QuickDraw10

This dataset was collected by Google from people drawing different objects. The dataset is a collection of 50 million drawings from 345 different objects that is available publically for everyone. We extract a simple subset of the dataset for simple machine learning tasks. The dataset is suggested as an alternative for MNIST. 

![alt text](https://raw.githubusercontent.com/zaidalyafeai/QuickDraw10/master/images/qd-img.png)
![alt text](https://raw.githubusercontent.com/zaidalyafeai/QuickDraw10/master/images/qd-tsne.png)

## Alternative for MNIST
MNIST has many disadvantages

*   No great variablitiy in the data for each class. The number of strokes is limited for each drawn digit. 
*  Most MNIST pairs could be distinguised by just one pixel. See [this](https://gist.github.com/dgrtwo/aaef94ecc6a60cd50322c0054cc04478). 
*  The data is overused in both universities and the literature. 

## Get the Data

| Name  | Content | Examples | Size | Link|
| --- | --- |--- | --- |--- |
| `train-ubyte.npz`  | training set images and labels  | 80,000|21 MBytes | [Download](https://github.com/zaidalyafeai/QuickDraw10/blob/master/dataset/train-ubyte.npz?raw=true)|
| `test-ubyte.npz`  | testing set images and labels  | 20,000|6  MBytes | [Download](https://github.com/zaidalyafeai/QuickDraw10/blob/master/dataset/test-ubyte.npz?raw=true)|

Alternatively, you can clone this GitHub repository; the dataset appears under `dataset/`.

## Labels
Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | Cloud|
| 1 | Sun |
| 2 | Pants |
| 3 | Umbrella |
| 4 | Table |
| 5 | Ladder |
| 6 | Eyeglasses |
| 7 | Clock |
| 8 | Scissors |
| 9 | Cup|

## Loading data with Python (requires [NumPy](http://www.numpy.org/))

```python
import numpy as np

train_data = np.load('dataset/train-ubyte.npz')
test_data  = np.load('dataset/test-ubyte.npz')

x_train, y_train = train_data['a'], train_data['b']
x_test,  y_test  = test_data['a'],  test_data['b']
```
