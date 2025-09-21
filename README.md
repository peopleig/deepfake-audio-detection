# deepfake-audio-detection
An ML model written in Python, which detects AI/Machine generated audio using ResNet34

## What each .py file does:  
### augment.py  
Defines audio data augmentation. This is used during model training to make the model robust  
  
### dataset.py  
Loads the ASVspoof dataset's audio files, and applies the augmentation defines in augment.py.  
This also prepares data loaders  
  
### features.py  
This will extract the audio features - LFCC or spectrograms - from the raw waveforms.  
  
### model.py  
This defines the ResNet model with a classifier head like softmax  

### train.py  
This is the main script which will train the model. This will handle the dataset, the feature extraction. 

### evaluation.py  
This will load a trained model checkpoint, and will run evaluation/testing - data accuracy and other stuff.  
  