# Uke 44

# Veileder møte


# Enviorment

# AI: Imitation

# AI: Natural selection

Tok i bruk kode som ble brukt for environment. 
Det erstattet mye av min kode (hente alle mulige actions før DOWN).
Ser ut som noe var feil med denne koden da det kjørte som forventet nå.

Kjørte 100 kandidater. De startet med gjennomsnittlig score 20
men etter bare 17 epoker med å lage og erstatte 30% nye kandidater, fikk vi en gjennomsnittlig score på 80

![log](./imgs/letris.gif)

Etter ~60 epoker hadde jeg vekter som ble ganske greit. De fikk 300 highscore på tetris og mer enn tusen på lettris.

Vi fikk tilgang til maskin på AI Lab som har mange flere kjerner så jeg implementerte threading for å øke hastigheten for treningen.
