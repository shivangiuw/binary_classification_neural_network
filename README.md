# Deep Neural network to predict Startup Success

Creating a binary classification model using  deep neural networks with different optimization techniques to predict whether startups will be successful if funded by Alphabet Soup (a venture capital firm )using dataset provided.

## Dataset:
The dataset used is a CSV file containing a variety of information of more than 34,000 organizations that have received funding from Alphabet Soup over the years, about each business, including whether or not it ultimately became successful. 
The dataset information columns can be categorized as below for building the neural network model:



Column Names (Features → X) |  Column Names (Target variable → y) |
----------------------------|-------------------------------------|
EIN                         |   IS_SUCCESSFUL                     |
NAME                        |                                     |        
APPLICATION_TYPE            |                                     |            
AFFILIATION                 |                                     |   
ORGANIZATION                |                                     |
USE_CASE                    |                                     |
STATUS                      |                                     |
INCOME_AMT                  |                                     |
SPECIAL_CONSIDERATIONS      |                                     |
ASK_AMT                     |                                     |


## Neural Network Model and Optimization attempts: 

Following steps were followed to build a binary classification neural network model:

 ### 1. Preprocess data for a neural network model

 * Dropping irrelevant variable columns i.e. "EIN" and "NAME"
 * Encoding categorical variables using `OneHotCoder`.
 * Creating the features`(X)` and target `(y)` dataset.
 * Split the X and y dataset into train and test data using `test_train_split`.
 * Scale the features datasets using `StandardScaler`.

### 2. Using the model-fit-predict pattern to compile and evaluate a binary classification model.

  * Create a deep neural network by assigning the `number of input features`, the `number of layers`,
  the `number of neurons` on each layer using Tensorflow’s Keras,the `relu` activation function for hidden
  layers and `sigmoid` activation function for outer layer.
  * Compile and fit the model `(nn)`using `the binary_crossentropy loss function`, the `adam optimizer`, 
   and the`accuracy evaluation` metric.  
   * Evaluate the model using the test data to determine the model’s loss and accuracy.


### 3. Optimize the neural network model.

   To improve on first model’s predictive accuracy, we will try three models with different optimization 
   techniques as following :
   
####   1. Add more hidden layers.
   *When designing a deep neural network, it is best to start with two hidden layers. Then, continue adding
   additional layers until the model’s performance no longer improves over the same number of
   epochs.*
   * Here,the original model was created with two hidden layers and now the new model `nn_A1` is tried with
   three hidden layers wherein the model was run with 50 and then 100 epochs. The model performed best at
   50 epochs and the performance dropped with 100 epochs.
   
####   2. Reduce the features
   Adjust the input data by dropping different features columns to ensure that no variables or outliers
   confuse the model.
   * During preprocessing of data for the original model two columns i.e. "EIN", "NAME" were dropped as 
   they do not  have any effect on performance and success of a startup. Now for the optimization of the 
   model, we have dropped two more columns i.e. "STATUS" and "SPECIAL_CONSIDERATIONS" as they seem to have
   less relevance and may confuse the model. The categorical variable columns of the reduced features data
   are then encoded and further all features scaled and used to create and fit a new network model `nn_A2`.
   
####   3. Add more neurons (nodes) to a hidden layer.
   *The total number of neurons across all hidden layers should be ⅔ the size of the input layer (size of 
   input layer = number of features), plus the size of the output layer (size of output layer = number of
   neurons on the output layer).Alternatively, the total number of neurons across all hidden layers should
   be less than twice the size of the number of features in the input layer.*
   * The third optimization model `nn_A3` was created with addition of two nodes(neurons) to all the three
   hidden layers which was then run with 50, 100 and 150 epochs where it performed best at 150 epochs.
   
   
 ### Summary of Results:
 
  The results of the performances of the original model as well as optimization models with metrics are as
  follows:
  
 Model | No.of Input Features| No.of Hidden Layers| No.of nodes in Layers|Epochs(best performance)|Loss%| Accuracy|
------ |---------------------|--------------------|----------------------|------------------------|-----|---------|
nn     |           116       |   2                | H1-58, H2-29         |        50              |55.72| 73.13   |
nn_A1  |           116       |   3                | H1-58, H2-29, H3- 15 |        50              |55.96| 73.06   |
nn_A2  |           113       |   3                | H1-57, H2- 29, H3-15 |        50              |55.26| 73.00   |
nn_A3  |           113       |   3                | H1-59, H2- 32, H3-18 |       150              |57.20| 72.93   |


- The best performance of the model was observed with the original model`(nn)` with highest accuracy of 73.13 as compared  to other models at 50 epochs which reduced with the increase in number of epochs.

- After increasing the number of hidden layers from 2 to 3 in the new neural network model`(nn_A1)` for optimization, the accuracy scores did not improve much at 50 epochs or further dropped on increasing the number of epochs.

- Further trying the other optimization method, the number of features were dropped from 116 to 113 to reduce outliers which may confuse the model, and new model `(nn_A2)` was tried with epochs which gave accuracy  and loss scores approximately same as previous two models at 50 epochs which seemed to be dropping with number of epochs. 

- Lastly, the new model`(nn_A3)` was tried with increased number of nodes in each layer which gave the highest performance metrics at 150 epochs at built model, however the accuracy scores and loss scores have dropped with increase in number of nodes.

None of the optimization techniques seem to have improved the performance of the models. However more trials can be done with dropping different features or increasing or decreasing number of nodes and hidden layers with different combinations and running the models at diffrent epochs to get accuracy score close to 1.

*All the 4 models(originals and optimization models) were run for 50, 100 and 150 epochs respectively, but the models have been saved as .h5 files only for the epochs at which best accuracy scores were observed at give number of layers and nodes.*


## Technologies and Modules

This tool leverages python 3.7 with the following packages:

* [pandas] (https://pandas.pydata.org/docs/getting_started/index.html)- for data analysis
* [scikit-learn] (https://scikit-learn.org/stable/)- open source machine learning library
* [pathlib] (https://docs.python.org/3/library/pathlib.html#module-pathlib)- to read file path
* [tensorflow] (https://www.tensorflow.org/api_docs/python/tf/keras)- open source platform for machine learning


## Installation Guide

```
conda install pandas
pip install --upgrade tensorflow
conda install -c intel scikit-learn

```

## Contributor

Shivangi Gupta

## License

MIT





