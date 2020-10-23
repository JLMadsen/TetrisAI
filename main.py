from enviorment.tetris import Tetris

# Manual testing of tetris env

env = Tetris()

def main():

    while 1:
        env.reset()
        done = False
        while not done:
            done = env.render(1)

if __name__ == "__main__":
    main()
