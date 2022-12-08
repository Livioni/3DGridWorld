import gym
env = gym.make("GridWorld-v0")
observation = env.reset()
# for _ in range(1000):
#        action = 0  # User-defined policy function
#        observation, reward, terminated, info = env.step(action)
#        env.my_render()
#        if terminated:
#               observation = env.reset()
env.my_render()
for i in range(8):
       action = 3
       observation, reward, terminated, info = env.step(action)
env.my_render()
for i in range(1):
       action = 5
       observation, reward, terminated, info = env.step(action)
env.my_render()
print(reward)
# for i in range(2):
#        action = 1
#        observation, reward, terminated, info = env.step(action)
# print(reward)
# env.my_render()


env.close()