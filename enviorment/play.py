from actions import Action
from tetris import Tetris

# Manual testing of tetris env

def main():
    
    env = Tetris()
    env.reset()
    env.render()

    action_map = {'a': Action.LEFT, 's': Action.DOWN, 'd': Action.RIGHT, 'w': Action.ROTATE, ' ': Action.WAIT}
    while 1:

        user_input = input('step:')

        action = 0
        try:
            action = action_map[user_input[0]]
        except:
            return

        env.step(action)
        env.render()



if __name__ == "__main__":
    main()
