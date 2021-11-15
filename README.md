# EEG-Multi-Tasking
 
 • **Abstract**  
Motor imagination is an act of thinking about body motor parts and motor imagery signals are the brain activities generated while performing motor imagination. Electroencephalogram (EEG) is a non- invasive way by which brain activities can be recorded with electrodes placed on the scalp. EEG measurements recorded for every other individ- ual are different even if they think about same moving body parts. And this makes motor imagery signal classification difficult. Many authors have proposed various machine learning and deep learning approaches for motor imagery signal classification. To the best knowledge of authors, in most of the studies, subject specific model is trained as EEG signature is subject specific. However, even when EEG measurments from different individuals are recorded, if they are for same motor imagination, then, there must be some hidden common features for a specific motor imagi- nation across all individuals. With subject specific models, those features can’t be learnt. We have proposed yet another deep learning approach for motor imagery signal classification where a single deep learning sys- tem is trained with a multi-task learning approach. The results illustrate that a single mult-task learnt model performs even better than subject specific trained models.    
**Keywords** : EEG · Motor Imagery · Deep Learning · Multi-task Learning.


  
![](Assets/Brain.jpg)

## Folders/Files Description
This repo contains following folders and files :  
1. input : Contains the input EEG spectral images
2. notebooks : Contains notebooks made while developing this project
3. models  : Pytorch .pth model files are stored in this directory 
4. results    : Results and training configs are stored as json file in this directory for each training
5. src    :  It contains all the python scripts i.e., data precossing, models, training and evaluation scripts.
6. main.py     : It is the main driver scripts which can be used for training each models.


## Training Steps
- First setup the project via : 
```bash
git clone https://github.com/RahulnKumar/EEG-Multi-Tasking.git
pip3 install -r requirements.txt
```
- For training and evaluation run __main__.py with these flags:

  Single task training :`python3 main.py --st`  

  Conventional Multi-task training : `python3 main.py --cmt`  

  Private-Shared Multitask training: `python3 main.py --psmt`  

  Adversarial Multitask training : `python3 main.py --amt`  
  

## Dataset details  
****


### BCI IV Dataset 2

#### Graz Dataset B  [(EEG Data with Feedback, 2 Class Classification)](http://bbci.de/competition/iv/desc_2b.pdf)  
This data set also consists of EEG data from 9 subjects. The cue-based BCI paradigm consisted of two different motor imagery tasks, namely the imagination of right hand and left hand.For each subject 5 sessions are provided, whereby the first two sessions contain training data without feedback (screening), and the last three sessions were recorded with feedback.  
Summary :    
  EEG data (3 + 3 channels, 250Hz) of 9 candidates and 2 classes     
  Total sessions = 5   
  1 session      = 6 runs seperated by short breaks    
  1 run          = 10 trials per class  
  Total trials   = (10 trials)*(2 classes)*(6 runs) = 120 trials per session   

   

## References

-----------


## Contributors  
 - Rahul Kumar and Sriparna Saha
 ## License & copyright
 © 2021 Rahul Kumar   
 Licensed under the [MIT License](LICENSE)
