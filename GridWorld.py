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
        self.border = 11 #设定边界范围 
        self.x_scaled_axis = np.linspace(-2,2,self.border)
        self.y_scaled_axis = np.linspace(-2,2,self.border)
        self.scaled_z = np.zeros([self.border,self.border],dtype=np.float32)
        for i in range(len(self.x_scaled_axis)):#计算peaks高度
            for j in range(len(self.y_scaled_axis)):
                self.scaled_z[i][j] = peaks(self.x_scaled_axis[i],self.y_scaled_axis[j])
        self.z = np.round(1 * (self.scaled_z-np.min(self.scaled_z))) #乘倍数放缩然后四舍五入
        
        #petrol spot
        self.petrol_x = np.array([5],dtype=np.int16)#设定1个充电桩
        self.petrol_y = np.array([5],dtype=np.int16)
        self.petrol_z = np.zeros_like(self.petrol_x)
        for i in range(len(self.petrol_x)):
            self.petrol_z[i] = self.z[self.petrol_x[i]][self.petrol_y[i]] + 1  #在地形上面一个点
        #terminal spot 
        self.terminal_x = np.array([3,5],dtype=np.int16)#设定4个需要观察的点
        self.terminal_y = np.array([4,6],dtype=np.int16)
        self.terminal_z = np.zeros_like(self.terminal_x)
        for i in range(len(self.terminal_x)):
            self.terminal_z[i] = self.z[self.terminal_x[i]][self.terminal_y[i]] + 1 #在地形上面一个点
        
        self.x_loc = None
        self.crashed = 0
        self.x_trace = []
        self.y_trace = []
        self.z_trace = []
        self.terminal_mask = [0,0] #用于记录已经探索到的观察点
        self.low_observation = np.zeros(18)
        self.high_observation = self.border * np.ones(18)
        self.observation_space = spaces.Box(low=self.low_observation, high=self.high_observation,dtype=np.int32)
        self.action_space = spaces.Discrete(6)

    def step(self,action):
        #6个动作分别为 上下左右前后
        obs = self._check_obs(self.x_loc,self.y_loc,self.z_loc)#检查动作是否为有效动作，即有没有触碰地形
        if obs[action] == 1:#如果触碰地形
            self.steps += 1
            # self.state = np.hstack((self.x_loc,self.y_loc,self.z_loc,self.obs))
            self.state = np.hstack((self.x_loc,self.y_loc,self.z_loc,self.obs))
            self.done = 0
            self.crashed += 1
            self.reward = -2
            return self.state, self.reward, self.done, {}
        elif action == 0:
            self.z_loc += 1
            # print("向上")
        elif action == 1:
            self.z_loc -= 1
            # print("向下")
        elif action == 2:
            self.y_loc -= 1
            # print("向左")
        elif action == 3:
            self.y_loc += 1
            # print("向右")
        elif action == 4:
            self.x_loc -= 1
            # print("向前")
        elif action == 5:
            self.x_loc += 1
            # print("向后")
        else:
            raise "wrong action"
        self.steps += 1
        self.x_trace.append(self.x_loc)
        self.y_trace.append(self.y_loc)
        self.z_trace.append(self.z_loc)
        self.obs = self._check_obs(self.x_loc,self.y_loc,self.z_loc)
        self.reward = self._check_reward(self.x_loc,self.y_loc,self.z_loc)  
        self.done = self._check_done(self.x_loc,self.y_loc,self.z_loc) 
        self.state = np.hstack((self.x_loc,self.y_loc,self.z_loc,self.obs))
        return self.state, self.reward, self.done, {} 

    def _check_reward(self,x,y,z):
        reward = -1
        for i in range(len(self.terminal_x)):
            if self.terminal_mask[i] == 1:
                continue
            else:
                if x==self.terminal_x[i] and y==self.terminal_y[i] and z==self.terminal_z[i]:
                    reward = 1000
                    print("Hit terminal spot %d" % i)
                    self.terminal_mask[i] = 1
                    return reward
        for i in range(len(self.petrol_x)):
            if x==self.petrol_x[i] and y==self.petrol_y[i] and z==self.petrol_z[i]:
                reward = 100
                print('End')
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
        if y-1 < 0:
            obs[2] = 1
        elif self.z[x][y-1] >= z:
            obs[2] = 1
        if y+1 >= self.border:
            obs[3] = 1
        elif self.z[x][y+1] >= z:
            obs[3] = 1
        if x-1 < 0:
            obs[4] = 1
        elif self.z[x-1][y] >= z:
            obs[4] = 1
        if x+1 >= self.border:
            obs[5] = 1
        elif self.z[x+1][y] >= z:
            obs[5] = 1 
        return obs 

    def reset(self):
        #随机初始点
        self.x_loc = random.randint(0,self.border-1)
        self.y_loc = random.randint(0,self.border-1)
        self.z_loc = random.randint(self.z[self.x_loc][self.y_loc],np.max(self.z))
        # self.x_loc = 8
        # self.y_loc = 5
        # self.z_loc = int(self.z[self.x_loc][self.y_loc]+1)
        
        self.x_trace = []
        self.y_trace = []
        self.z_trace = []
        self.terminal_mask = [0,0,0,0]
        self.steps = 0
        self.obs = self._check_obs(self.x_loc,self.y_loc,self.z_loc)
        self.state = np.hstack((self.x_loc,self.y_loc,self.z_loc,self.obs))
        return self.state

    def my_render(self):
        x = np.linspace(0,self.border-1,self.border)
        y = np.linspace(0,self.border-1,self.border)
        X, Y = np.meshgrid(x, y)
        # Plot the surface
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        #base map
        ls = LightSource(270, 45)
        rgb = ls.shade(self.z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(Y, X, self.z, rstride=1, cstride=1, facecolors=rgb,
                            linewidth=0, antialiased=False, alpha=0.6,shade=False)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        #scatter              
        colors = ('r') 
        ax.scatter(self.petrol_x, self.petrol_y, self.petrol_z, s=36, c=colors)
        colorsr = ('b','b')
        ax.scatter(self.terminal_x, self.terminal_y, self.terminal_z, s=36, c=colorsr)
        if self.x_loc is not None:
            ax.scatter(self.x_loc,self.y_loc,self.z_loc,s=40,c='k')
            ax.plot(self.x_trace,self.y_trace,self.z_trace)
        ax.view_init(elev=30., azim=-130)
        # ax.view_init(elev=90., azim=0)
        plt.show()
        # plt.draw()
        # plt.pause(0.001)  
        # plt.close()  

# grid = GridEnv()
# grid.my_render()