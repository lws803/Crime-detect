# Crime-detect  
### Crime and aggression detection  
  
## Introduction  
Crime-detect is a surveillance program designed to reduce the burden of threat detection on security personnel operating in the field of surveillance. We plan to use Optical Flow to detect levels of high movement in the frame. From there, we will raise a detection alert if there is a knife in the frame. Our plan is to implement two detectors for two of the most common weapons used in today’s crime scenes - pistols and knives.

![comic strip](https://github.com/lws803/Crime-detect/tree/master/images/Crimedetect_Comic_RedBox.png "Situation Comic")

## Problems Detected
- OpenPose creates problems when the knife detector is integrated, hard to distinguish between humans in the frame and who is holding the knife. Looking for other alternatives now. 

## Problem we are trying to solve
Did you know that there is a 1 in 2,499 lifetime chance of being involved in a stabbing? Moreover, can you believe In the United States there is a 1 in 315 chance of being part of a shooting? Now imagine, how often such encounters can be detected and resolved in a quick and efficient manner, thereby potentially saving lives? With encounters such as this occurring often, and with an increasing trend of mass shootings, a smart surveillance system that can instantly alert the authorities of violent crime is vital in ensuring the safety of the general public and our loved ones.

## The solution we built 
Our current implementation has 2 detection functions, one for detecting crimes committed when there’s little movement (e.g. armed assault or robbery) and another for crimes committed where there is high movement (eg. stabbing). To avoid false detections, we have implemented a voting system where stale detections are cleared and only when recent detections surpass a certain value, then it will send an alert to the operator. We also have an intuitive GUI which displays the feeds and a special tab to store the images of the detected events.

## Is our solution creative? We believe so!
Other crime prediction software, such as Cloud Walk, perform citizen trustworthiness analyses based on the data obtained from institutions such as police, hospitals, schools, banks and possibly from social media. However, we believe that such a system can potentially be prone to discrimination against certain situations that may not involve any crime. In our solution, there is no social credit system that keeps a permanent log of all activity, thus there is no sense of constant monitoring and assessment. Instead, Machine Learning is used to detect the potential crime and react instantly to the scenario by alerting the user or relevant authorities.

## So how does it work?
The following flowchart describes the steps of our program in detecting crime:
