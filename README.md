<p align="center">
  <img src="./rapporter/imgs/tetrisAI.png" height=150 />
</p>

Full report in norwegian <a href="https://denlurevind.com/content/dqn.pdf">here.</a>

<p align="center">
  <img src="https://i.imgur.com/zZnZpdI.gif"/>
</p>

## Genetic algorithm vs Imitation + DQN

<p align="center">
  <img src="./rapporter/imgs/comparison1.png" height=300 />
  <img src="./rapporter/imgs/comparison2.png" height=300 />
</p>

# Usage

`main.py` for playing Tetris manually.

`main_dqn.py` for training and testing DQN.

`main_imitation.py` for training and data collection Imitation learning.

`main_natselect.py` for training and testing Natural Selection

`benchmark.py` for comparing agents.

`demo.py` for running multiple agents.

## Enviorment
To edit Tetris behaviour change contructor params.

```py
env = Tetris({
  'reduced_shapes': True,
  'reduced_grid': True
})
```

Enviorment follows regular OpenAI standard.
```py
from enviorment.tetris import Tetris

from Imitation.agent import imitation_agent
from nat_selection.model import Model
from dqn.agent import DQN

env = Tetris() 

# Agent can either be nat_select model, DQN, Imitation, or custom.
# agent = imitation_agent(env)
# agent = Model()
agent = DQN(env)

total_score = 0
state, reward, done, info = env.reset()

while not done:

  # for random action
  action = env.action_sample 
  
  # for agent action
  # Be aware that different agents have different methods of getting the next action.
  # example: https://github.com/JLMadsen/TetrisAI/blob/95ef54f92eb04ee3ac6f0664e823ef4a8bab932e/benchmark.py#L85.
  action = agent.policy(state)
  
  state, reward, done, info = env.step(action)
  total_score += reward
  
  env.render()
```

State is a three dimensional array with 2 layers representing the placed blocks and the ones you are controlling. Blocks are 1 and blank cells are 0.

<p align="center">
<img src="https://i.imgur.com/wtMRG0E.png">
</p>

Loading pretrained models can be done like this.
```py
from enviorment.tetris import Tetris

from Imitation.agent import imitation_agent
from nat_selection.model import Model
from dqn.agent import DQN

env = Tetris() 

#DQN
agent = DQN(env)
agent1.load_weights('_60k_3') 
# Where the argument is the suffix after "weights" in ./dqn/weights

# Imitation
agent = imitation_agent(env)
agent.load_weights('_10k_01_nat1')
# Where the argument is the suffix after "weights" in ./Imitation/weights

# Natural selection (Genetic algorithm)
agent = Model([-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306])
#              sum of heights       cleared lines        holes                evenness
# Where each number is the individual weight.
```

# Install

> pip install -r requirements.txt

and <a href="https://pytorch.org/" target="_blank">Pytorch</a>

