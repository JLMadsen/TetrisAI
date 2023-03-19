"""
Runs until max_score is reached
frames are saved in rapporter/recordings
run `ffmpeg -framerate 25 -i %04d.jpeg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4`
in recordings folder to create video
"""

from enviorment.tetris import Tetris
from nat_selection.agent import Agent as NatAgent
from nat_selection.model import Model
import pygame, os

try: os.makedirs("rapporter/recordings")
except OSError: pass

env = Tetris()
candidate = Model([-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306])

def main():
    # record until given score
    max_score = 30
    file_num = 0
    score = 0

    while score < max_score:
        state, reward, done, info = env.reset()

        while score < max_score:

            action = candidate.best(env)

            for a in action:
                env.render()
                state, reward, done, info = env.step(a)
                score += reward
                file_num += 1
                pygame.image.save(env.screen, f"rapporter/recordings/{str(file_num).rjust(4, '0')}.jpeg")

if __name__ == "__main__":
    main()
