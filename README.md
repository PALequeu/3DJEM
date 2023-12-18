# 3DJEM
Final project for the Topics In Machine Intelligence class of Seoul National Univeristy by Kang Seung-Yeop, Lequeu Pierre-Antoine and Hellum Jacob.
This project applies the Joint Energy-Based (JEM) framework to a 3D classifier (PointMLP) in order to work on generation tasks from a classification model.


To start the training:
```
python train_wrn_ebm.py —lr .0001 —optimizer adam —p_x_weight 1.0 —p_y_given_x_weight 0.1 —p_x_y_weight 0.0 —sigma .03 —width 10 —depth 28 —save_dir ./experiment/ —plot_uncond —warmup_iters 1000
```
