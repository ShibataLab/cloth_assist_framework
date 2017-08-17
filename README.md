# cloth_assist_interface

Implementation of reinforcement learning framework for robotic clothing assistance on baxter research robot.

## Run BGPLVM Latent Space Baxter controller for clothing assistance task
1. Open terminal and enable robot
```
$ cd ros_ws
$ ./baxter.sh
$ rosrun baxter_tools enable_robot.py -e
$ rosrun baxter_examples gripper_cuff_control.py
```
2. Open new terminal and start kinect
```
$ ./baxter.sh
$ roslaunch kinect2_bridge kinect2_bridge.launch publish_tf:=True
```
3. Open new terminal and start puppet mode
```
$ ./baxter.sh
$ rosrun cloth_assist_framework teach.py -l right
```
4. Move baxter arms to mannequin and attach grippers to T-shirt eyelets. Then stop puppet mode by pressing `Ctrl+C` in the `teach.py` terminal.
5. Start the latent space controller for the demo
```
$ roslaunch cloth_assist_framework model_player.launch file:=/home/baxterpc/clothing/demo/model.p
```
6. Setup the windows such that left half is Rviz and right upper half is kinect and right lower half displays latent space.
7. Use this setup to control baxter robot through latent space.
8. Close all terminals and switch off baxter after the demo.

## Design BGPLVM model
In the procedure mentioned above, BGPLVM model i.e. `model.p` file is needed (Step 5 in _Run BGPLVM Latent Space Baxter controller for clothing assistance task_) in order to control the robot. This sections explains the steps to design this model file.
1. Open new terminal and start puppet mode
```
$ ./baxter.sh
$ rosrun cloth_assist_framework teach.py -l right
```
2. Attach t-shirt to both the grippers
3. Teach a trajectory and record it to csv file
```
$ rosrun  cloth_assist_framework record.py -f teach -m 1
```
4. The trajectory may need some pre-processing, such as slicing the inital and last sample points from trajecotry and re-sampling the trajecotry.
```
$ rosrun  cloth_assist_framework trajectory.py -f teachJA -t0 1 -t1 15 -s procJA -n 200
```
In the above step, we ignored the samples before 1 seconds and after 15 seconds from input trajecotry file `teachJA`. This trajecotry is then down-sampled to 200 samples and saved to trajecotry file `procJA`
5. Train a BGPLVM model on the processed trajecotry file
```
rosrun cloth_assist_framework model_creator.py -f procJA -s model
```
6. The above step may not produce smooth latent space sometimes. Please rerun the `model_creator.py` and limit the number of traning iterations as following:
```
rosrun cloth_assist_framework model_creator.py -f procJA -s model -n 20
```

## Real-time tracking of Clothing article from Kinect Data
1. Open terminal and start kinect
```
$ ./baxter.sh
$ roslaunch kinect2_bridge kinect2_bridge.launch publish_tf:=True
```
2. Run cloth tracker node
```
$ rosrun  cloth_assist_framework cloth_tracker ../calib
```
3. A window is going to pop-up from the above step. Please select the cloth by left clicking and then dragging the mouse over the cloth article. The selected rectangular region can be seen highlighted in this window.
4. Press `q` to finish the initilization. A new window is going to show segmented clothing article in real-time
