OBSERVATION_PROMPT = "Start by stating what effect your action had, was it what you wanted? Why? Did the board change? Then reply with a few sentences of plain-text strategy observation about the frame to inform your next action."

SYSTEM_INSTRUCTIONS = """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.
"""

CHOOSE_ACTION_PROMPT = """Call a tool defining an action."""