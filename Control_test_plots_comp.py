import numpy as np
from DDQN_UNIF_torch import DDQNAgent
from SpaceDEnv_original import AttitudeControlEnv
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

class Agent_Env(object):
    def __init__(self, Ename) -> None:
        frameskip = 5
        self.env = AttitudeControlEnv(frameskip=frameskip, steps=1000)
        self.ddqn_agent = DDQNAgent(lr=0.0001, gamma=0.99,  max_epsilon=0,eps_end=0, epsilon_decay=5e-7,
        batch_size=64, input_dims=7,fc1_dims=256, fc2_dims=256, n_actions=19,replace_target=500,name=Ename)

        self.ddqn_agent.load_models()
    
    def Genarate_traj(self, observation):
        Q = observation[0:4]
        Quat_init = np.array([Q[1], Q[2], Q[3], Q[0]])
        observation = self.env.reset(Quat_init)
        states = [np.array(observation)]   
        done = False
        while not done:
            action = self.ddqn_agent.choose_action(observation)
            observation_new,_,done,_= self.env.step(action)
            states.append(np.array(observation_new))
            observation = observation_new
        return states

Ename1 = 'Spacecraft_DDQN_UNIF_2'
Ename2 = 'Spacecraft_DDQN_UNIF_4'

frameskip = 5
env = AttitudeControlEnv(frameskip=frameskip, steps=700)
observation = env.reset()

states = Agent_Env(Ename=Ename1).Genarate_traj(observation=observation)
states2 = Agent_Env(Ename=Ename2).Genarate_traj(observation=observation)

print(np.linalg.norm(states[-1][4:]))
print(np.linalg.norm(states2[-1][4:]))

x = np.linspace(0, len(states)*frameskip//240, len(states))

q4 = [i[0] for i in states]
q1 = [i[1] for i in states]
q2 = [i[2] for i in states]
q3 = [i[3] for i in states]

q4_2 = [i[0] for i in states2]
q1_2 = [i[1] for i in states2]
q2_2 = [i[2] for i in states2]
q3_2 = [i[3] for i in states2]

w1 = [i[4] for i in states]
w2 = [i[5] for i in states]
w3 = [i[6] for i in states]

w1_2 = [i[4] for i in states2]
w2_2 = [i[5] for i in states2]
w3_2 = [i[6] for i in states2]

fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                subplot_titles=("q4", "q1", "q2", "q3"))

fig.add_trace(go.Scatter(x=x, y=q4, mode='lines', name='q4'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=q4_2, mode='lines', name='q4_2'), row=1, col=1)

fig.add_trace(go.Scatter(x=x, y=q1, mode='lines', name='q1'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=q1_2, mode='lines', name='q1_2'), row=2, col=1)

fig.add_trace(go.Scatter(x=x, y=q2, mode='lines', name='q2'), row=3, col=1)
fig.add_trace(go.Scatter(x=x, y=q2_2, mode='lines', name='q2_2'), row=3, col=1)

fig.add_trace(go.Scatter(x=x, y=q3, mode='lines', name='q3'), row=4, col=1)
fig.add_trace(go.Scatter(x=x, y=q3_2, mode='lines', name='q3_2'), row=4, col=1)


fig.update_layout(title_text="Angular Orientation over time", 
                font=dict(family="Arial", size=25, color="#000000"),
                plot_bgcolor='white')

fig.update_xaxes(title_text="Timestep (sec)", tickfont=dict(size=15),
                gridcolor='grey')
fig.update_yaxes(title_text="Quaternions", tickfont=dict(size=15), 
                gridcolor='grey')

fig.show()



fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

fig.add_trace(go.Scatter(x=x, y=w1, mode='lines', name='wx'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=w1_2, mode='lines', name='wx_2'), row=1, col=1)

fig.add_trace(go.Scatter(x=x, y=w2, mode='lines', name='wy'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=w2_2, mode='lines', name='wy_2'), row=2, col=1)

fig.add_trace(go.Scatter(x=x, y=w3, mode='lines', name='wz'), row=3, col=1)
fig.add_trace(go.Scatter(x=x, y=w3_2, mode='lines', name='wz_2'), row=3, col=1)


fig.update_layout(title_text="Angular Velocity over time", 
                font=dict(family="Arial", size=25, color="#000000"),
                plot_bgcolor='white')

fig.update_xaxes(title_text="Timestep (sec)", tickfont=dict(size=15),
                gridcolor='grey')
fig.update_yaxes(title_text="Ang Velocity", tickfont=dict(size=15), 
                gridcolor='grey')

fig.show()