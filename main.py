import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
ENV = 'CartPole-v0'


def build_model():
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse',
                  optimizer=Adam(lr=0.01))
    return model

def main():
    model = build_model()
    sample = [[0]*1000000,[0]*1000000,[0]*1000000]
    index = 0
    env = gym.make(ENV)
    action = env.action_space.n
    print(env.get_state())
    s = env.reset()
    index = 0
    begin_index = 0
    R = 0
    while index<1000000:
        act_values =model.predict(np.reshape(s,[1,4]))
        a = np.argmax(act_values[0])
        sample[0][index] = a
        sample[1][index] = s
        s_, r, done, info = env.step(a)
        R = R + 1
        if done:  # terminal state
            print(index-begin_index)
            s = env.reset()
            for i in range(begin_index, index + 1):
                sample[2][i] = R
                target_f = model.predict(np.reshape(sample[1][i],[1,4]))
                target_f[0][sample[0][i]] = R
                model.fit(np.reshape(s,[1,4]), target_f, epochs=1, verbose=0)
            begin_index = index + 1
            R = 0
        else:
            s = s_
        index = index + 1


if __name__ == "__main__":
    main()
