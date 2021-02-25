CE888 
project 1 
assignment 1
users guide

the mode variable controles the programs behaviour.

there are 3 main functions.

loading will expect a pre created model to be present in the appropriate directory as defined by the model name varaible. 

the datasets and model save locations are all fully configurable but also could result in unwanted overwrites so be carefull

the dataset is available at https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs

testing the loaded model will display an accuracy score

training the loaded model will train against the defined dataset, then create a save of the new model in the defined directory

generate is used when no model is yet present and will create a set architecture. before saveing in the defined directory.

the current model "Fire_detect_prototype_V2" is currently in this git repo.
