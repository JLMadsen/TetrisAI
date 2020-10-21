from actions import Action
from tetris import Tetris

# Manual testing of tetris env

def main():
    
    env = Tetris()
    action_map = {'a': Action.LEFT, 's': Action.DOWN, 'd': Action.RIGHT, 'w': Action.ROTATE, ' ': Action.WAIT}
    
    while 1:
        env.reset()
        env.render()
        done = False
        
        while not done:

            user_input = input('step:')

            action = 0
            try:
                action = action_map[user_input[0]]
            except:
                return

            observation, reward, done, info = env.step(action)
            if done:
                print('Done')
            env.render()



if __name__ == "__main__":
    main()
