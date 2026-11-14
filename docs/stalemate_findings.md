# Stalemate and opponent-strength findings

Measured with `scripts/run_experiments.py` using teams without an LLM, 30 games per pairing
(15 seeds, commander team on both sides), global observability, randomised near-base spawns, 200-turn limit.

| Team | vs aggressive | vs balanced | vs defensive | vs camper | vs classical | vs random | vs stay |
|---|---|---|---|---|---|---|---|
| classical (Team 1) | 30W | 30W | 30W | 25D / 5L | 14W / 14L / 2D | 30W | 29W / 1D |
| scripted (role defaults) | 30W | 30D | 30D | 30D | 4W / 26D | 17W / 13D | 15W / 15D |

(W = win, D = draw at the turn limit, L = loss.)

## Findings
1. **A defender standing on its own base cell makes that base unbeatable.** Pickup is resolved before tagging
   in the same turn, so every pickup there is tagged immediately. Team 1's classical AI wins 0 of 30 against the
   `camper` opponent (25 draws, 5 losses to the camper's attacker). Both the RL branch (`ON_FLAG` guard mode)
   and Team 1's defender can produce this situation.
2. **The simple role-default team draws against every opponent that keeps a defender near its flag**
   (balanced, defensive, camper): 30 of 30 draws each.
3. **The provisional `aggressive`, `balanced` and `defensive` opponents are too weak for comparisons.**
   Team 1's classical AI beats each of them in 30 of 30 games, so they cannot separate stronger commander
   variants. The `classical` opponent is the only one that gives close results (14-14-2).

## Decisions needed from the group
- Rules: keep as is, or change how pickup/tagging interacts with a defender on the base cell
  (for example, tagging not applied on the flag-pickup turn, or a minimum distance for defenders).
- Opponent set for the evaluation: which definitions of aggressive / defensive / balanced, and whether the
  main comparisons use the classical AI as the opponent.
