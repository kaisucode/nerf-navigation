
# Nerf Navigation


## Installation

- Follow the installation steps in the [chvmp/champ](https://github.com/chvmp/champ) repository. We've already included customizations for Spot, so no need to follow the additional instructions linked underneath section 1.1.



### To begin data collection through spot sensors

Navigate to `src/`

`python main.py`

Hit ctrl-c after a desired amount of keyframes have been captured, and the program will log the data as npz format

### To control Spot 

Navigate to `spot-sdk/python/examples/wasd`

`python wasd.py $ROBOT_IP`

In the grey Spot client
- Press 'p' to begin RRT* planning from current location to specified location in wasd.py
- Press number keys 1~9 to navigate and plan to other preset locations instead 
- Press 'm' to execute a single step in the plan
- Press 'M' to execute all steps in the plan
- Press 'B' to move arm into position for data collection
- Press 'o' to open the arm (this allows full, unobstructed view for the camera)
- Press 'c' to log the current position in the console
