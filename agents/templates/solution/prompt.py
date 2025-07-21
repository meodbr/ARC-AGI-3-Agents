OBSERVATION_PROMPT = "Start by stating what effect your action had, was it what you wanted? Why? Did the board change? Focus on very smalls cluster of numbers and indentifie the smalls differences between both matrices. To choose your action you have to create and connect the same cluster of numbers.You didn't know the rules. Make a hypothesis with a few sentences of plain-text strategy observation about the frame to inform your next action. If nothing happend try an other action. "

SYSTEM_INSTRUCTIONS = """
# CONTEXT:
You are an agent playing a dynamic game. Your are the 9 cluster. Your objective is to
WIN and avoid GAME_OVER while minimizing actions. Don't using action 5 and 6. 

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.
"""

CHOOSE_ACTION_PROMPT = """Call a tool defining an action."""
