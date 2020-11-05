# Uke 44

# Veileder møte

* Samme nettverk i DQN og Imitation
* Blokken må gå ned automatisk etter en hvis tid
* Kjøre samme rekkefølge av brikker
* Parrallellprossesering
* Leke med parameterene i nat-select
* Begynne på rapport

# Enviorment

# AI: Imitation

Fikk satt opp et nettverk som kjørte og trente. Fikk problem med at den satt seg fast enten helt til høyre eller helt til venstre. Men etter veileder møtet, la vi inn at den gikk ned av seg selv, og da ble det fikset. Måtte da skaffe ny data å trene på. Så blir det å utforske litt på learning rate og få den til å spele.

# AI: Natural selection

Tok i bruk kode som ble brukt for environment. 
Det erstattet mye av min kode (hente alle mulige actions før DOWN).
Ser ut som noe var feil med denne koden da det kjørte som forventet nå.

Kjørte 100 kandidater. De startet med gjennomsnittlig score 20
men etter bare 17 epocher med å lage og erstatte 30% nye kandidater, fikk vi en gjennomsnittlig score på 80

![log](./imgs/letris.gif)
