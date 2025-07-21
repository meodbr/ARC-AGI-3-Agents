from arc_agi_babies.config.settings import settings


SYSTEM_INSTRUCTIONS_GENERIC = """Based on the examples INPUT and OUTPUT pixel grids given, infer the TEST OUTPUT pixel grid."""

SYSTEM_INSTRUCTIONS_TASK_UNKNOWN = """Based on the examples INPUT and OUTPUT pixel grids given, infer the TEST OUTPUT pixel grid.
The rule to get from input to output is:
each 1 is a blue pixel
each 2 is a red pixel
There are uncompleted blue (1) squares in the input, they are each missing one blue (1) pixel to be complete, let's call it the EXIT.

In the output:
* Each blue (1) square must be filled by red (2) pixels.
* The EXIT is pointing in a direction, a red (2) line must be drawn from this exit to the opposite side of the grid.
"""

SYSTEM_INSTRUCTIONS_TASK_007BBFB7 = """Based on the examples INPUT and OUTPUT pixel grids given, infer the TEST OUTPUT pixel grid.
The INPUT is a 3x3 matrix representing a pattern of 0 and another number
To make the OUTPUT:
The output is a 9x9 matrix, you should reproduce the pattern at this scale but with a specificity
Each 0 in the INPUT matrix is scaled by 3 and becomes a 3x3 filled entirely with 0.
Each number in the INPUT matrix is becoming a 3x3 matrix representing exactly the INPUT pattern.

To explain you visually how the problem is working, I mapped every INPUT pixel to the area it influences:
INPUT:
a b c
d e f
g h i
OUTPUT:
[
 [a, a, a,  b, b, b,  c, c, c],
 [a, a, a,  b, b, b,  c, c, c],
 [a, a, a,  b, b, b,  c, c, c],

 [d, d, d,  e, e, e,  f, f, f],
 [d, d, d,  e, e, e,  f, f, f],
 [d, d, d,  e, e, e,  f, f, f],

 [g, g, g,  h, h, h,  i, i, i],
 [g, g, g,  h, h, h,  i, i, i],
 [g, g, g,  h, h, h,  i, i, i],
]

repeat Exactly the INPUT matrix in the non-zero areas
"""

SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_TASK_007BBFB7