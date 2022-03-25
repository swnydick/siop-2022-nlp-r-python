#########################################
# Python and R Setup (Using Python)     #
#                                       #
# Korn Ferry Institute: Automation Team #
# 2022-04-29                            #
#########################################

# 1. Setup =====================================================================

# need to first make sure we are using the correct conda environment
# bash/zsh/fish: ~% conda activate $HOME/Library/r-miniconda/envs/r-siop-nlp

# then need to install the rpy2 package (see 1-python_and_r_setup.R)
# bash/zsh/fish: ~% conda install rpy2

# then need to indicate directory of project
project_dir = "siop-2022-nlp-r-python"
outer_dir   = "/Users/nydicks/Documents/Projects/Workshops and Training/Workshops/2022"

# note: do not use rpy2 through a session where you are using reticulate, as it
#       will crash!! only use rpy2 in a standard python session!

# 2. Setup rpy2 ================================================================

# importing other objects
import os

# change directory to workshop location
os.chdir(os.path.join(project_dir, outer_dir))

# importing the necessary objects
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages   as rpackages # import packages
import rpy2.robjects.conversion as rpyconv
from rpy2.robjects.conversion import rpy2py, py2rpy
from rpy2.robjects import numpy2ri

# connecting with the utils package and setting CRAN mirror for installation
utils = rpackages.importr("utils")

# where are we connecting to the library?
base  = rpackages.importr("base")

# -> leading "." gets translated to "_"
base._libPaths().r_repr()

# -> change library path to current project
base._libPaths(new = "renv/library/R-4.1/x86_64-apple-darwin17.0")
base._libPaths().r_repr()

# -> seeing what packages we have installed (good, although very confusing to read)
utils.installed_packages().r_repr()

# note: DON'T install packages this way simply because they won't be maintained
#       by renv!

# 3. Basic Usage ===============================================================

## a. R Objects ----------------------------------------------------------------

# pull out object from R
pi_r   = robjects.r("pi")

# but it's still an R object, so you need to convert it back to python
import numpy as np
pi_py  = np.array(pi_r) # OR rpy2py(pi_r)

# create object in R (can use "", but convention to use """)
robjects.r("""mat_r <- matrix(1:10, nrow = 2)""")
mat_r  = robjects.globalenv["mat_r"]
mat_r

# convert it into numpy array
mat_py = np.array(mat_r)
mat_py

# you can also use rpyconv.activate and rpyconv.deactivate to automatically
# control this behavior!

## b. Python Objects -----------------------------------------------------------

# need to activate conversation (won't work with numpy otherwise)
numpy2ri.activate()

# can create object in python
vec_py = np.array(range(5))

# and then assign them back to R
robjects.r.assign("vec_r", vec_py)

# we can then pull them out of the global environment
robjects.globalenv["vec_r"]

# if you try to represent the object in R, it will crash without deactivating
# robjects.globalenv["vec_r"].r_repr() # doesn't work

numpy2ri.deactivate()
robjects.globalenv["vec_r"].r_repr() # now works (is IN R!!!)

## c. Indexing Arrays ----------------------------------------------------------

# you can index these arrays the same way you would index regular python arrays
# (even if they haven't been converted to Python yet)
vec_py = robjects.globalenv["vec_r"]

vec_py[0]    # first element
vec_py[2:3]  # third element
vec_py[2:]   # third element to last element
vec_py[:-1]  # all but last element
vec_py[::-1] # reverse

# for matrices, they are treated like R vectors with classes
mat_r[0:3]   # the first 3 elements
mat_r[0, 1]  # doesn't work
mat_py[0, 1] # works!

## d. Functions ----------------------------------------------------------------

# we can create functions in R by running R code
robjects.r('''add <- function(x, y) x + y''')
add_r  = robjects.r["add"]

# we can run objects from python into this function
add_r(2, 3)           # can add scalars
add_r(vec_py, 3)      # can add python vectors and scalars
add_r(vec_py, vec_py) # can add python vectors together
add_r(mat_r, mat_r)   # this doesn't work ... until

numpy2ri.activate()   # we activate conversion
add_r(mat_r, mat_r)   # but then it automatically converts the output to python

numpy2ri.deactivate()

# note: old versions would require you to manually convert objects back and
#       forth (e.g., ri2py and py2ri), but they don't seem to always work
#       (at least the same way that they used to)
# note: you can turn on automatic conversion to and from np using numpy2ri.activate()

# See https://rpy2.github.io/doc/latest/html/introduction.html for more capabilities
