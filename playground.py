import gym
env = gym.make("GridWorld-v0")
observation = env.reset()
for _ in range(1000):
       action = 0  # User-defined policy function
       observation, reward, terminated, info = env.step(action)
       env.my_render()
       if terminated:
              observation = env.reset()
env.close()