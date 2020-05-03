import gym
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

env = gym.make('FrozenLake-v0')       # is_slippery True
env.render()                            # Show the initial board
state = env.reset()

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
