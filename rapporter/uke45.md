# Uke 44

# Veileder møte

* Samme nettverk i DQN og Imitation
* Blokken må gå ned automatisk etter en hvis tid
* Kjøre samme rekkefølge av brikker
* Parrallellprossesering
* Leke med parameterene i nat-select
* Begynne på rapport

# Enviorment

Implementert metoder for å lagre state slik at agentene kan simulere trekk.

```py
checkpoint = self.save_checkpoint()

# do stuff

self.load_checkpoint(checkpoint)

```

# AI: DQN

For å trene det nevrale nettverket lager vi en replay buffer hvor man kan hente ut tidligere trekk

```py
batch = self.memory.sample(batch_size)

for state, action, next_state, reward in batch:
    
    # learn
```

# AI: Imitation

Fikk satt opp et nettverk som kjørte og trente. Fikk problem med at den satt seg fast enten helt til høyre eller helt til venstre. Men etter veileder møtet, la vi inn at den gikk ned av seg selv, og da ble det fikset. Måtte da skaffe ny data å trene på. Så blir det å utforske litt på learning rate og få den til å spele.

# AI: Natural selection

Tok i bruk kode som ble brukt for environment. 
Det erstattet mye av min kode (hente alle mulige actions før DOWN).
Ser ut som noe var feil med denne koden da det kjørte som forventet nå.

Kjørte 100 kandidater. De startet med gjennomsnittlig score 20
men etter bare 17 epoker med å lage og erstatte 30% nye kandidater, fikk vi en gjennomsnittlig score på 80

![log](./imgs/letris.gif)

Etter ~60 epoker hadde jeg vekter som ble ganske greit. De fikk 300 highscore på tetris og mer enn tusen på lettris.

Vi fikk tilgang til maskin på AI Lab som har mange flere kjerner så jeg implementerte threading for å øke hastigheten for treningen.
