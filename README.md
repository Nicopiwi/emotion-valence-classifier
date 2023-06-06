# Emotion Valence Classfication from EEG data- Brainhack School Project 2023
----

## Introduction
Emotion classification within the realm of Brain-Computer Interaction (BCI) presents an intriguing area of study. The ability to accurately identify and understand human emotions through brain activity has the potential to revolutionise various fields, including healthcare, gaming, and mental health. Recent advances in Artificial Intelligence (AI) and Machine Learning (ML) have opened up new avenues for analysing and decoding emotions from EEG (Electroencephalography) data. By leveraging the power of AI and ML, we aim to gain deeper insights into the neural correlates of emotions and develop an emotion classification models.

## Dataset
For this purpose, we use the Imagined Emotion Dataset from OpenNeuro. This dataset provided us with EEG recordings obtained from subjects while they experienced different emotional states. By using this dataset, we aimed to build a model capable of accurately predicting emotional valence based on the recorded brain activity.

## Models 
To achieve our objective, we chose two distinct models: the Random Forest Classifier, a traditional ML technique, and EEGNet, a Convolutional Neural Network (CNN) approach utilising Deep Learning principles. By comparing the performance of these models, we sought to evaluate their efficacy in classifying emotional valence.
**Preprocessing**
We first identified the emotions present in our dataset, enabling us to isolate the relevant data segments for analysis. Subsequently, we divided the data into epochs based on these emotional events, allowing us to focus specifically on periods associated with specific emotions. Afterwards, for each model we have subsequently processed the data to fit the model specification.

## Results 

## Conclusions

## Deliverables
- a python file to preprocess the data
- Two notebooks with the model implementation

**Notes**
- In order to replicalte the results, download the dataset from OpenNeuro and put it into `./dataset/` directory and then create a preprocessed_data directory so that the preprocessed data can be saved into that path. 
- Please download the EEGNet Model implementation and pretrained weights from this [https://github.com/vlawhern/arl-eegmodels/tree/master](link).
- 


## References

**Dataset**
- Julie Onton and Scott Makeig (2022). Imagined Emotion Study. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds003004.v1.1.1
