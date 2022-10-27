# URL
Repo for final project at Reinforcement Learning class at my Institute '22

## Background 
We want to use robot (Universal Robots UR10, for example) to insert craging cable into corresponding socket.  To do this, we need to align camera principal axis with socket. We do this by marking 6 points on socket's surface and minimizing the distance between points' projections on camera plane and desired projections positions, which we have calculated and hardcoded.

## Requirements
- numpy
- scipy
- opencv-python
- torch
- tqdm
- matplotlib

## Contents
<b>RL_env</b> contains the main pipeline of model's training<br>
<b> rl_socket/agent.py </b> contains RL Agent class with correspondong methods<br>
<b> rl_socket/camera_transition.py </b> contains a class for camera simulation, with camera-only params like intrincic matrix, and methods to project 3D points to camera plane<br>
<b> rl_socket/test.ipynb </b> is a notebook with tests of model<br>

## Authors
| Autor | github | telegram | 
| --- | --- | --- |
| Petr Sokerin | [petrsokerin](https://www.github.com/petrsokerin)  | [petrsokerin](t.me/petrsokerin) | 
| Andrey Puchkov | you are here, but [still](https://www.github.com/andpuc23) | [snipercapt](t.me/snipercapt) | 
| Nikita Belousov | [nokiroki](https://www.github.com/nokiroki) | [nokiroki1](t.me/nokiroki1) | 
