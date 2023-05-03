# DeepRL_Attitude_Control

Programming language and  Libraries used :
1. Python 3.7.11
2. torch==1.11.0+cpu
3. numba==0.56.4
4. numpy==1.21.6
5. gym==0.26.2
6. plotly==5.14.1

Sources used as template :
1. https://colab.research.google.com/github/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb#scrollTo=qEx9h3n0uVUK
2. https://github.com/jakeelkins/rl-attitude-control
3. https://youtu.be/wc-FxNENg9U

File descriptions and  Instructions :
1. DDQN_PER_torch.py and DDQN_UNIF_torch.py files are where DDQN network, Experience replay and action selection code is written (followed youtube tutorial on simple DQN from source 3 for UNIF case and borrowed code from source 1 & made changes as per my requirement and convinience ) .
2. SpaceDEnv_original.py is the environment file borrowed from Source 2  and made changes in reward function and state defination.
3. SPCMain_UNIF.py and SPCMain_UNIF.py main files which should be executed for training a model and it's corresponding average returns over episodes  is stored in the current directory by executing Plot_Training_res.py file .
4. ddqn directory have all the model parameters files which can be loaded and tested for a random initial states by executing Control_test_plots.py.
