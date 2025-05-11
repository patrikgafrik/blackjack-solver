# Riešenie hry Blackjack pomocou biologických algoritmov

Tento projekt obsahuje dva prístupy k hľadaniu optimálnej stratégie pre hru Blackjack pomocou prírodou inšpirovaných algoritmov:

- **Genetický algoritmus** (`genetic_algorithm.py`)
- **Neurónová sieť** (`neural_network.py`)

Obe metódy využívajú prostredie `Blackjack-v1` z knižnice Gymnasium na simuláciu a vyhodnotenie stratégií.

---

### Reprezentácia stavu
Stav hry je reprezentovaný ako trojica `(player_sum, dealer_upcard, usable_ace)`:
- **player_sum**: Súčet hodnôt kariet hráča (4-21)
- **dealer_upcard**: Viditeľná karta dealera (2-10, A)
- **usable_ace**: Indikátor, či hráč má použiteľné eso (0/1)

### Reprezentácia stratégie
Stratégia je typicky reprezentovaná dvomi maticami:

- **hard_totals**: 17x10 matica pre rozhodnutia s tvrdými súčtami (bez použiteľného esa)
    - Riadky: súčty kariet hráča od 4 po 20
    - Stĺpce: viditeľná karta dealera (2-10, A)
- **soft_totals**: 9x10 matica pre rozhodnutia s mäkkými súčtami (s použiteľným esom)
    - Riadky: súčty kariet hráča od 12 po 20
    - Stĺpce: viditeľná karta dealera (2-10, A)

### Akcie (hodnoty v maticiach)
- `0`: **STAND** (zostať stáť)
- `1`: **HIT** (ťahať ďalšiu kartu)

### Vyhodnotenie stratégie
Všetky tri metódy simulujú hry s použitím danej stratégie a merajú priemerný zisk:
- **Výhra**: +1
- **Remíza**: 0
- **Prehra**: -1