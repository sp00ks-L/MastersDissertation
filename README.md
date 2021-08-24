# MastersDissertation
 A repository containing the code to support my Masters dissertation. The dissertation involved implementing an alternative method of ChIP-seq analysis which is a popular genome-wide technique to identify protein-DNA interactions. I chose to implement a convolutional neural network (CNN) to identify regions of DNA from the fission yeast <i>Schizosaccharomyces pombe</i> to which a regulatory protein (Rif1) binds. In essence, this became a binary classifier; given a new image of the ChIP data, does the image contain a putative binding site or not.
 
### Data Generation
Contains the scripts that were used to label and generate the data for my convolutional neural network as well as some helper functions. 

### Extra
Contains additional code that I wrote to visualise the data and the predictions made by my model. You can check these plots online at [ChIP-seq Data](http://lukejones.co.uk/chip_plot.html) and [Model Predictions](http://lukejones.co.uk/chip_predictions.html) respectively.

### Making Predictions
Here are all the scripts that were used to produce the data that is used to make the final predictions from the saved model as well as creating additonal data specifically to label genomic regions with. The ```load_img_data.py``` file in this directory is based of a file with the same name in the Models directory - it is however tailormade to be used with the prediction data - it does not split into test, train, valiation etc.

### Models
This directory contains my Keras models. Both ```third model.py``` and ```final_model.py``` are the same model, ```third model.py``` just contains plotting functionality to assess performance whereas ```final_model.py``` just creates the labels and confidence scores. I have also included an extra file to illustrate how I used hyperband for hypertuning. 

 
## Example Use
 1. Use ```Data Labelling.py``` to label origin regions of interest
 2. Use ```Chromo sampler.py``` to produce the train, test and validation images for the CNN
 3. Use ```final_model.py``` to train on the data and save the trained model to file
 4. Use ```Images2Predict.py``` to produce the images for the chromosome you're interested in labelling. I primarily used Chromosome I of <i>
Schizosaccharomyces pombe</i>.
 5. Use ```model_multi_predict.py``` to create <i>n</i> predictions. (Currently setting <i>n</i> >= 1 does not provide any advantage)
 6. Either
    - Use ```plot_predictions.py``` to see a simple plot of the results 
   OR
    - Use ```predictions with origins.py``` for a more advanced webplot using Plotly
 

## Further Updates
Despite having completed my dissertation, I will probably update the code slightly to make it a single tool rather than the individual execution of different scripts. The file paths currently are in a format for use on Windows and will need re-doing if a person wanted to use this tool on unix.
