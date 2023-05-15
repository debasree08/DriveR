
# <strong><em>Which Self-driving Mode Should I Enable on Your Streets? Auto-annotating Road Segments based on Driving Safety</em></strong>
Although fully autonomous vehicles are in their rising stage, understanding the road infrastructure and surrounding traffic is
necessary to decide their autonomy level. A road segment where frequent traffic rules violations occur may not be suitable
for a higher level of autonomy. In contrast, a well-maintained road segment with sparse traffic can definitely incorporate full
autonomy. Thus, understanding the safety of a road segment is crucial for the further implementation of driving automation.
DriveR proposes a novel approach to automatically annotate the road segments based on their driving safety to inform
the self-driving mode that should be enabled on these segments. For this purpose, our system sought out the causal chain
responsible behind poor driving maneuvers whenever there is a degradation in driving behavior. Eventually, we propose a
novel metric safety score derived from the severity and frequency of such causal chains faced by a set of vehicles passing
through a road junction. Finally, each road junction is annotated with the safety score in a Likert scale of [1 − 5], where 1
denotes least safe and 5 denotes most safe. We have thoroughly evaluated DriveR using both in-house dataset and a public
dataset UAH-DriveSet comprising 33 & 8.33 hours of driving data, obtaining a F1-score of 0.81. Overall, this paper offers an
important contribution to the field of digital map annotation concerning driving safety and provides a foundation for further
research in this area.
Use the following commands in sequence to run the system.
<!-- <p align="center">
      <img src="Images/framework.png" width="70%"/>
</p> -->

## In-house Dataset
We primarily utilize the IMU, the GPS, and the video data captured through the front camera (facing towards the front windscreen) as different modalities. A sample dataset is provided with the following modalities which is collected over three different cities in develpping country. The cities are named as city1, city2 and city3. For each city, we have provided 20 minutes of data, cumulating upto 1 hour. Due to large file size, we have provided the dataset in the following link.

https://drive.google.com/drive/folders/1yK-jDdpKFq0Ts1xxYUkvTy1DKTZa9Dit?usp=sharing


The GPS data is in the following format:
- LOCATION Latitude :;LOCATION Longitude :; YYYY-MO-DD HH-MI-SS_SSS;

The IMU data is in the following format:
- timestamp | G:y-axis,z-axis,x-axis



| Modalities    | Sampling Rate/fps | Duration (in seconds) |
|---------------|-------------------|-----------------------|
| Accelerometer | 15                | 60                    |
| GPS           | 1                 | 60                    |
| Video         | 30                | 60                    |

## Data Annotation Platform
- We have annotated our dataset using this platform developed by us: https://drivingskillsvalidator.web.app/

## User Interface Web Application
- Our auto-annotated digital map can be viewed through this platform developed by us: https://drivingskillsmap.glitch.me/


## Commands to run the model
- python data_collect.py <file_name> // Collect Data
- python maneuvers.py <file_name> // Feature Extraction 
- python script_vid.py <file_name> // Preprocessing
- Under the Model subdirectory, find the .ipynb file as VI_RL notebook  // Running the model
## Dataset
A sample in-house dataset is provided in the dataset folder
https://drive.google.com/drive/folders/1yK-jDdpKFq0Ts1xxYUkvTy1DKTZa9Dit?usp=sharing
<!-- # Reference
To refer the <em>DriveR</em> framework or the dataset, please cite the following work.

<!-- [Download the paper from here](https://dl.acm.org/doi/10.1145/3549548). -->

<!-- BibTex Reference:
```
@INPROCEEDINGS{}

```
For questions and general feedback, contact Debasree Das (debasree1994@gmail.com). --> 

