# DeepRL_Attitude_Control

Programming language and  Libraries used :
1. Python 3.7.11
2. torch==1.11.0+cpu
3. numba==0.56.4
4. numpy==1.21.6
5. gym==0.26.2
6. plotly==5.14.1

Use pyenv for maintaining python virtual environment : https://realpython.com/intro-to-pyenv/

Sources used as template :
1. https://colab.research.google.com/github/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb#scrollTo=qEx9h3n0uVUK
2. https://github.com/jakeelkins/rl-attitude-control
3. https://youtu.be/wc-FxNENg9U

File descriptions and  Instructions :
1. DDQN_PER_torch.py and DDQN_UNIF_torch.py files are where DDQN network, Experience replay mechanism   code is written (followed youtube tutorial on simple DQN from  source 3 and made some changes in DQN & Agent classes for UNIF case | For PER case i borrowed code from source 1 & made changes as per requirement in the agent class and some class methods like compute loss ) .
2. SpaceDEnv_original.py is the environment file borrowed from Source 2  and made changes in reward function and state defination.
3. SPCMain_UNIF.py and SPCMain_PER.py main files which should be executed for training a model and it's corresponding average returns over episodes  is stored in the current directory by executing Plot_Training_res.py file ( Written almost by me).
4. ddqn directory have all the model parameters files which can be loaded and tested for a random initial states by executing Control_test_plots.py for getting the state trajectory and  Control_test_plots_comp.py for comparing between different models (written almost by me).


Incase of any problem reach out to me : luckykn@iitk.ac.in
