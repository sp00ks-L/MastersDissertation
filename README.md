# MastersDissertation
 A repository containing the code to support my masters dissertation
 
## Data Generation
Contains the scripts that were used to label and generate the data for my convolutional neural network

## Extra
Contains additonal code that I wrote to visualise the data and the predictions made by my model. You can check these plots online at [ChIP-seq Data](http://lukejones.co.uk/chip_plot.html) and [Model Predictions](http://lukejones.co.uk/chip_predictions.html) respectively

## Making Predictions
Here are all the scripts that were used to produce the data that we make the final predictions, as well as the various scripts to make predictions from the saved model. 
The <load_img_data.py> file in this directory is based of a file with the same name in the Models directory - it is however tailormade to be used with the prediction data - it does not split into test, train, valiation etc

## Models
This directory contains my Keras models. Both <third model.py> and <final_model.py> are the same model, <third model.py> just contains plotting functionality to assess performance whereas <final_model.py> just creates the labels and confidence scores. I have also included an extra file to illusrtate how I used hyperband for hypertuning. 
