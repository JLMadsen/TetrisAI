# Uke 43

Starter med å dele opp i 2 brancher, enviorment og modell slik at vi kan lage spillet og modellen samtidig.
Modellen bruker openAI gym mens spillet blir laget.

# Enviorment

Implementerer Tetris etter openAI standarden.

- Step
- Reset
- Render

![mvp](./imgs/tetris.png)

# AI: Genetisk algoritme

Begynte å lage ai som er basert på "natural selection":
- Lager x initielle kandidater
- For de 2 beste av et tilfeldig 10% blir det laget 30% nye kandidater
- Disse 30% erstatter de verste 30% av kandidatene

Kandidatene kjører gjennom alle mulige steps før DOWN og velger den beste basert på vekter.
Disse vekter er tilfeldig i starten men skal konvergere når kandidatene begynner å bli utveklset.
Fikk den til å kjøre med det var masse bugs som førte til at det løste oppgaven.
