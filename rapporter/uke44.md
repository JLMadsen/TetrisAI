# Uke 44

# Veileder møte

Møte med Jonathan 27.10.

Snakket om hva slags reinforcement learning vi burde fokusere på fremover.

- Imitation Learning
    - Lærer på vår spilling
- Deep Q Learning
    - Utforsker spilling selv
    - Kan fores på imitation learning vekter

En stor ulempe med DQN er tiden det tar å trene.

Derfor skal vi implementere en letter versjon av Tetris slik at vi kan sjekke modellen med kortere trening.

# Enviorment

Vi møter på en bug hvor når man roterer en brikke hender det at den flytter seg et steg nedover. Vi tror dette er pga måten brikke data er lagret og hentes.

Issue: https://github.com/JLMadsen/TetrisAI/issues/5

Implementerte config slik at det er enkelt å endre på ting

I Tetris klassen

![config](./imgs/config.png)

I main

![config](./imgs/config2.png)

# AI: DQN

Startet med å modellere det nevrale nettverket til DQN modellen.

Vi forholder oss til en enkel modell så lenge den fungerer.

```py
self.q_net = nn.Sequential(
    nn.Conv2d(2, 32, (20, 10)),
    nn.ReLU(),
    nn.Conv2d(32, 64, (1, 1)),
    nn.ReLU(),
    nn.Linear(1, env.action_space)
)
```

Prøver også ut forskjellige metoder for å gi en "score" basert på hvor brikken blir plassert.

# AI: Imitation

Startet arbeider med en imitation agent, satt først opp metoder for å lage og lese data til agenten.

Begynte deretter å implementere selve agenten. Satt bare opp grunn strukturen. Neste uke må det jobbes med selve nettverket slik at det kan begynne å kjøre og faktisk trene

# AI: Genetisk algoritme

Løste mange problemer som hadde med utregning av verdier for valg av beste move å gjøre.
Meste av bugs hadde med indexer å gjøre da noen random funksjoner og prosentregninger endte opp med å være out of bounds.
Likevel er det fortsatt noen problemer som fører til at det blir valgt feil trekk. 
Har eksperimentert med forskellige testparametere men det er fortsatt en eller flere bugs som ødelegger valg av move.
Det er ganske sikkert at feilen ligger innen henting av alle trekk, da jeg får riktige verdier under manuell spilling.



