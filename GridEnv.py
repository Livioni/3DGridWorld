import numpy as np
import gym,random
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource

def peaks(x,y): 
    z = 3*(1-x)**2*np.exp(-x**2-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-1/3*np.exp(-(x+1)**2-y**2)
    return z

class GridEnv(gym.Env):
    def __init__(self):
        self.x_scaled_axis = np.linspace(-2.4,2.4,49)
        self.y_scaled_axis = np.linspace(-2.4,2.4,49)
        self.scaled_z = np.zeros([49,49],dtype=np.float32)
        for i in range(len(self.x_scaled_axis)):
            for j in range(len(self.y_scaled_axis)):
                self.scaled_z[i][j] = peaks(self.x_scaled_axis[i],self.y_scaled_axis[j])
        self.z = np.round(10 * (self.scaled_z-np.min(self.scaled_z)))

        #petrol spot
        self.petrol_x = np.array([8,20,25,40],dtype=np.int16)
        self.petrol_y = np.array([10,15,32,30],dtype=np.int16)
        self.petrol_z = np.zeros_like(self.petrol_x)
        for i in range(len(self.petrol_x)):
            self.petrol_z[i] = self.z[self.petrol_x[i]][self.petrol_y[i]] + 1  
        #terminal spot
        self.terminal_x = np.array([25,17,36,44],dtype=np.int16)
        self.terminal_y = np.array([30,9,21,39],dtype=np.int16)
        self.terminal_z = np.zeros_like(self.terminal_x)
        for i in range(len(self.terminal_x)):
            self.terminal_z[i] = self.z[self.terminal_x[i]][self.terminal_y[i]] + 1 

        self.terminal_mask = [0,0,0,0]
        self.low_observation = np.zeros(10)
        self.high_observation = np.max(self.z) * np.ones(10)
        self.observation_space = spaces.Box(low=self.low_observation, high=self.high_observation,dtype=np.int32)
        self.action_space = spaces.Discrete(6)

    def step(self,action):
        #上下左右前后
        obs = self._check_obs(self.x_loc,self.y_loc,self.z_loc)
        if obs[action] == 1:
            self.steps += 1
            self.state = np.hstack((self.x_loc,self.y_loc,self.z_loc,self.obs,self.steps))
            self.done = 0
            self.reward = -1
            return self.state, self.reward, self.done, {}
        elif action == 0:
            self.z_loc += 1
        elif action == 1:
            self.z_loc -= 1
        elif action == 2:
            self.x_loc -= 1
        elif action == 3:
            self.x_loc += 1
        elif action == 4:
            self.y_loc -= 1
        elif action == 5:
            self.y_loc += 1
        else:
            raise "void action"
        self.steps += 1
        self.obs = self._check_obs(self.x_loc,self.y_loc,self.z_loc)
        self.reward = self._check_reward(self.x_loc,self.y_loc,self.z_loc)  
        self.done = self._check_done(self.x_loc,self.y_loc,self.z_loc) 
        self.state = np.hstack((self.x_loc,self.y_loc,self.z_loc,self.obs,self.steps))
        return self.state, self.reward, self.done, {} 

    def _check_reward(self,x,y,z):
        reward = -1
        for i in range(len(self.terminal_x)):
            if self.terminal_mask[i] == 1:
                continue
            else:
                if x==self.terminal_x[i] and y==self.terminal_y[i] and z==self.terminal_z[i]:
                    reward = 1000
                    self.terminal_mask[i] = 1
                    return reward
        for i in range(len(self.petrol_x)):
            if x==self.petrol_x[i] and y==self.petrol_y[i] and z==self.petrol_z[i]:
                reward = 500
                return reward
        return reward

    def _check_done(self,x,y,z):
        done = 0
        for i in range(len(self.petrol_x)):
            if x==self.petrol_x[i] and y==self.petrol_y[i] and z==self.petrol_z[i]:
                done = 1
                break
        return done

    def _check_obs(self,x,y,z):
        obs = np.zeros(6,dtype=np.int32)#上下左右前后
        if z+1 > np.max(self.z):
            obs[0] = 1
        if self.z[x][y] >= z-1:
            obs[1] = 1
        if x-1 < 0:
            obs[2] = 1
        elif self.z[x-1][y] >= z:
            obs[2] = 1
        if x+1 >= 49:
            obs[3] = 1
        elif self.z[x+1][y] >= z:
            obs[3] = 1
        if y-1 < 0:
            obs[4] = 1
        elif self.z[x][y-1] >= z:
            obs[4] = 1
        if y+1 >= 49:
            obs[5] = 1
        elif self.z[x][y+1] >= z:
            obs[5] = 1 
        return obs 

    def reset(self):
        #随机初始点
        # self.x_loc = random.randint(0,48)
        # self.y_loc = random.randint(0,48)
        # self.z_loc = random.randint(self.z[self.x_loc][self.y_loc],np.max(self.z))
        self.x_loc = 24
        self.y_loc = 24
        self.z_loc = int(self.z[self.x_loc][self.y_loc] + 2)
        self.steps = 0
        self.reward = 0
        self.done = 0
        self.obs = self._check_obs(self.x_loc,self.y_loc,self.z_loc)
        self.state = np.hstack((self.x_loc,self.y_loc,self.z_loc,self.obs,self.steps))
        return self.state

    def render(self):
        x = np.linspace(0,49,49)
        y = np.linspace(0,49,49)
        X, Y = np.meshgrid(x, y)
        # Plot the surface
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        #base map
        ls = LightSource(270, 45)
        rgb = ls.shade(self.z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(X, Y, self.z, rstride=1, cstride=1, facecolors=rgb,
                            linewidth=0, antialiased=False, alpha=0.6,shade=False)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        #scatter 
        petrol_x = np.array([8,20,25,40],dtype=np.int16)
        petrol_y = np.array([10,15,32,30],dtype=np.int16)
        petrol_z = np.zeros_like(petrol_x)
        for i in range(len(petrol_x)):
            petrol_z[i] = self.z[petrol_x[i]][petrol_y[i]] + 1                
        colors = ('r', 'r', 'r', 'r') 
        ax.scatter(petrol_x, petrol_y, petrol_z, s=36, c=colors)
        terminal_x = np.array([25,17,36,44],dtype=np.int16)
        terminal_y = np.array([30,9,21,39],dtype=np.int16)
        terminal_z = np.zeros_like(petrol_x)
        for i in range(len(petrol_x)):
            terminal_z[i] = self.z[terminal_x[i]][terminal_y[i]] + 1  
        colorsr = ('b','b','b','b')
        ax.scatter(terminal_x, terminal_y, terminal_z, s=36, c=colorsr)
        plt.show()

