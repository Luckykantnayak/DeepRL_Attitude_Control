import numpy as np

# For uniform sampling uncomment this
from DDQN_UNIF_torch import DDQNAgent

# For prioritized sampling uncomment this
# from DDQN_PER_torch import DDQNAgent

from SpaceDEnv_original import AttitudeControlEnv
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

if __name__ == '__main__':
    Ename = 'Spacecraft_DDQN_UNIF_2'  # change this to PER model file name present in ddqn directory
    I = np.array([[0.872, 0,0],[0, 0.115,0],[0,0, 0.797]])
    frameskip = 5
    env = AttitudeControlEnv(frameskip=frameskip, steps=700)
    
    ddqn_agent = DDQNAgent(lr=0.0001, gamma=0.99,  max_epsilon=0, epsilon_decay=5e-7,
    batch_size=64, input_dims=7,fc1_dims=256, fc2_dims=256, n_actions=19,replace_target=500,name=Ename)

    ddqn_agent.load_models()
    observation = env.reset()

    states = [np.array(observation)]   
    done = False
    while not done:
        action = ddqn_agent.choose_action(observation)
        observation_new,_,done,_= env.step(action)
        states.append(np.array(observation_new))
        observation = observation_new
    
    x = np.linspace(0, len(states)*frameskip//240, len(states))
    q4 = [i[0] for i in states]
    q1 = [i[1] for i in states]
    q2 = [i[2] for i in states]
    q3 = [i[3] for i in states]

    w1 = [i[4] for i in states]
    w2 = [i[5] for i in states]
    w3 = [i[6] for i in states]
    fig = go.Figure()
    
    print(np.linalg.norm(states[-1][4:]))

    fig.add_trace(go.Scatter(x=x, y=q4,
                        mode='lines',
                        name='q4'))
    fig.add_trace(go.Scatter(x=x, y=q1,
                        mode='lines',
                        name='q1'))
    fig.add_trace(go.Scatter(x=x, y=q2,
                        mode='lines',
                        name='q2'))
    fig.add_trace(go.Scatter(x=x, y=q3,
                        mode='lines',
                        name='q3'))

    # fig.add_trace(go.Scatter(x=x, y=reward_list,
    #                     mode='lines',
    #                     name='reward'))


    fig.update_layout(
    title={
        'text': "Angular Orientation over time",
        'x': 0.5,  # Set the x-position to 0.5 for center alignment
        'y': 0.95,  # Adjust the y-position if needed
        'xanchor': 'center',  # Set the x-anchor to center
        'yanchor': 'top',  # Set the y-anchor to top
        'font': {
            'family': "Arial",
            'size': 50,
            'color': "#000000"
        }
    },
    xaxis_title="Timestep (sec)",
    yaxis_title="Quaternions",
    font=dict(
            family="Arial",
            size=38,
            color="#000000"
        ),
    plot_bgcolor='white',
    xaxis=dict(
        gridcolor='grey',  # set the grid color for x-axis
        tickfont=dict(size=30),  # set the font size for x-axis tick labels
    ),
    yaxis=dict(
        gridcolor='grey',  # set the grid color for y-axis
        tickfont=dict(size=30),  # set the font size for y-axis tick labels
    ),
    #margin=dict(l=50, r=50, t=150, b=50),  # adjust the plot margin
    )


    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=w1,
                        mode='lines',
                        name='wx'))
    fig.add_trace(go.Scatter(x=x, y=w2,
                        mode='lines',
                        name='wy'))
    fig.add_trace(go.Scatter(x=x, y=w3,
                        mode='lines',
                        name='wz'))

    fig.update_layout(
        title={
        'text': "Angular Velocity over time",
        'x': 0.5,  # Set the x-position to 0.5 for center alignment
        'y': 0.95,  # Adjust the y-position if needed
        'xanchor': 'center',  # Set the x-anchor to center
        'yanchor': 'top',  # Set the y-anchor to top
        'font': {
            'family': "Arial",
            'size': 50,
            'color': "#000000"
        }
        },
        xaxis_title="Timestep (sec)",
        yaxis_title="Angular Velocity (rad/s)",
        font=dict(
            family="Arial",
            size=38,
            color="#000000"
        ),
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='grey',  # set the grid color for x-axis
            tickfont=dict(size=30),  # set the font size for x-axis tick labels
        ),
        yaxis=dict(
            gridcolor='grey',  # set the grid color for y-axis
            tickfont=dict(size=30),  # set the font size for y-axis tick labels
        ),
        #margin=dict(l=50, r=50, t=150, b=50),  # adjust the plot margin
    )

    fig.show()
    



    

   