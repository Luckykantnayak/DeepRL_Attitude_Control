
import numpy as np
from DDQN_PER_torch import *
from SpaceDEnv_original import AttitudeControlEnv
import time

if __name__ == '__main__':
    Ename = 'Spacecraft_DDQN_PER_2'
    I = np.array([[0.872, 0,0],[0, 0.115,0],[0,0, 0.797]])
    env = AttitudeControlEnv(steps=650)
    ddqn_agent = DDQNAgent(lr=0.0001, gamma=0.99,  max_epsilon=1.0, epsilon_decay=5e-7,
    batch_size=64, input_dims=7,fc1_dims=256, fc2_dims=256, n_actions=19,replace_target=500,name=Ename)
    #ddqn_agent.load_models()
    n_games = 6000
    
    ddqn_scores = []
    eps_history = []
    tic = time.time()
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        t = 0
        while not done:
            action = ddqn_agent.choose_action(observation)
            
            observation_,reward,done,_= env.step(action)
            score+= reward
            ddqn_agent.store_transition(observation, action, reward, observation_, done)
            ddqn_agent.train(i,n_games)
            observation = observation_
            

        eps_history = np.mean(ddqn_agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        print('episode', i, 'score %.2f' % score,'average score %.2f' % avg_score,'\n')
        #print('episode', i, 'score ' ,score,'average score ' ,avg_score)

        if i%10 == 0 and i>0:
            ddqn_agent.save_models()
    toc = time.time()
    print("Training time: ",(toc-tic)/60," min")  
    # filename = Ename+'.png'
    # x = [i+1 for i in range(n_games)]
    # plotLearning(x, ddqn_scores, filename)

    

    # Open a text file for writing
    with open(Ename+'.txt', 'w') as f:
        # Loop through the list of scores and write each score to the file
        for score in ddqn_scores:
            f.write(str(score) + '\n')

