# fklearn Examples

...

## Run batect (everything here...)
There are still some rough edges in there, but it works! If you want to
 run anything in here, see how to correctly install the dependencies,
 download the data, lint the files etc. you can use batect which has 
 it's little container where it already works just fine.
 
 
```
$ ./batect --list-tasks

Build tasks:
- get_data: download nlp data to local file
- run_example_1: compile the files to target

Utility tasks:
- dep_0: Download pipenv dependency & linter
- dep_1: Download dependencies
- dep_2: Download dev dependencies (run only if nec.)
- lint: lint python files
- shell: Open shell in container
```

in the meantime: use pipenv to install the right dependencies.

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
 and not like in the first case independent of it.

## Example 2 NLP Example (DONE)
Check out "2_nlp_example.py" to see how to use
fklearn for a NLP task, in particular:
- how to write a new curried function
- use the predefined lightGBM classifier

## Example 3 Grid Search (DONE)
Finally take a look at "3_grid_search.py" to see
- how to use the built in grid_search
- the built in n fold splitters & evaluators