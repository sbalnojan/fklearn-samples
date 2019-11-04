# fklearn Examples

...

## Run batect (everything here...)
(not running yet)
```
$ ./batect --list-tasks
```

## Example 1 (minimal example, DONE)
Check out the two files 1_easy.py and 1_easy_wo_functions.py and compare the code.

The real difference in the fklearn/the functional approach is one of 
the model of development design.

1. In the case of fklearn (see 1_easy.py) you have a dataframe (an object)
and apply a series of transformations/regressions/model-functions to it.
This is pretty important, the data itself considered an object,
but all of the transformations are applied as functions.
2. In the second case however, you'd usually think about the data as
having a "transform/load" method as part of the object (check out the code to
 see what I mean). That makes the methods dependent on the specific data,
 and not like in the first case indepdendent of it.

## Example 2 NLP Example (DONE)
Check out "2_nlp_example.py" to see how to use
fklearn for a NLP task, in particular:
- how to write a new curried function
- use the predefined lightGBM classifier

## Example 3 Grid Search (NOT DONE)
