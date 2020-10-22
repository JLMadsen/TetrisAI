from enviorment.tetris import Tetris

# Manual testing of tetris env

def main():
    
    env = Tetris()
    
    while 1:
        env.reset()
        
        done = False

        while not done:
            done = env.render(1)

if __name__ == "__main__":
    main()
