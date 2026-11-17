# Cortaí

Jogo de cortar frutas da região Norte (açaí, cupuaçu, pupunha, guaraná, bacuri, tucumã) feito para treinar uma IA que corta frutas e evita bombas.

## Rodar

```
npm start        # python -m http.server 8080 → http://localhost:8080
npm test         # checagem da lógica (corte, bomba, rótulos, seed)
```

## API para a IA (`window.game`)

- `game.slice({x,y}, {x,y})`: corta com um segmento (coordenadas do canvas 960×540)
- `game.entities`: frutas/bombas em jogo (ground truth, só para dataset/debug)
- `game.labels()`: caixas YOLO do frame atual (`0 = fruta`, `1 = bomba`)
- `game.reset(seed)`: mesma seed, mesma partida

## Gerar dataset

Clique em **Gerar dataset** (Chrome/Edge) e escolha uma pasta vazia. O jogo roda com um bot e salva
`images/{train,val}/*.jpg` e `labels/{train,val}/*.txt` no formato YOLO.
