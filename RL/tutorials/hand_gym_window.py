import gym
from gym.envs.registration import register
from colorama import init
import msvcrt

class _Getch:
    def __call__(self):
        keyy = msvcrt.getch()
        return msvcrt.getch()

init(autoreset=True)    # Reset the terminal mode to display ansi color
inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
arrow_keys = {
    b'H': UP,
    b'P': DOWN,
    b'M': RIGHT,
    b'K': LEFT
}

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')        # is_slippery False
env.render()                             # Show the initial board

while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print(arrow_keys[key])
        print("Game aborted!")
        break

    action = arrow_keys[key];
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break
