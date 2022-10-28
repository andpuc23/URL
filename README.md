# URL
Repo for final project at Reinforcement Learning class at my Institute '22

## Background 
We want to use robot (Universal Robots UR10, for example) to insert craging cable into corresponding socket.  To do this, we need to align camera principal axis with socket. We do this by marking 6 points on socket's surface and minimizing the distance between points' projections on camera plane and desired projections positions, which we have calculated and hardcoded.

## Requirements
```
numpy~=1.23
scipy==1.9.1
opencv-contrib-python~=4.6.0
torch==1.12.1
tqdm
matplotlib==3.5.3
```
## Contents
<b>RL_env.ipynb</b> contains the main pipeline of model's training<br>
<b>rl_socket/agent.py</b> contains RL Agent class with correspondong methods<br>
<b>rl_socket/camera_transition.py</b> contains a class for camera simulation, with camera-only params like intrincic matrix, and methods to project 3D points to camera plane<br>
<b>rl_socket/models.py</b> contains models for RL temporal difference<br>
<b>rl_socket/nn_models.py</b> contains neural networks behind Actor and Critic inside corresponding classes
<b>rl_socket/utils.py</b> is a bunch of utilitary methods for the project

## Authors
| Autor | github | telegram | role |
| --- | --- | --- | --- |
| Petr Sokerin | [petrsokerin](https://www.github.com/petrsokerin)  | [petrsokerin](t.me/petrsokerin) | project manager |
| Andrey Puchkov | you are here, but [still](https://www.github.com/andpuc23) | [snipercapt](t.me/snipercapt) | mathematician|
| Nikita Belousov | [nokiroki](https://www.github.com/nokiroki) | [nokiroki1](t.me/nokiroki1) | implementer |


### Project presentation
[Google.Drive link](https://docs.google.com/presentation/d/1-syFEiSdq753Tp182NRJFxDqdaC6vYLn/edit?usp=sharing&ouid=107167356583930004446&rtpof=true&sd=true)
