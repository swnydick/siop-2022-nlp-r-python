#########################################
# Python and R Setup (Using R)          #
#                                       #
# Korn Ferry Institute: Automation Team #
# 2022-04-29                            #
#########################################

# 1. Setup =====================================================================

# repo to use
options(repos = "https://cran.rstudio.com/")

# get project directory (SPECIFY MANUALLY???)
project_dir  <- here::here()
analyses_dir <- file.path(project_dir, "exercises")

setwd(project_dir)

# use renv to load the required packages
renv::restore()

# 2. Required Packages =========================================================

# if you don't have the latest version of each package, please install them
packageVersion("reticulate") # install.packages("reticulate")
packageVersion("tensorflow") # install.packages("tensorflow")
packageVersion("keras")      # install.packages("keras")

# we will always use the "pkg::fun()" syntax for reticulate for clarity.
median(c(1, 2, 3, 4))        # if a package is loaded, you can use its name
stats::median(c(1, 2, 3, 4)) # use the "::" syntax to not fully load the package

# remove RETICULATE_PYTHON environment variable to make things easier
Sys.unsetenv("RETICULATE_PYTHON")

# 3. Setup Reticulate ==========================================================

## a. Choose Environment -------------------------------------------------------

# If you do nothing? Then reticulate will go through a complicated algorithm to
# pick the "appropriate" version of python. This can be boiled down to ...
# (see https://github.com/rstudio/reticulate/blob/main/R/config.R)
# - 1. USE THE FIRST ONE OF THESE AUTOMATICALLY
#     a. Python already initialized
#     b. RETICULATE_PYTHON environment variable set
#     c. RETICULATE_PYTHON_ENV environment variable set
#     d. pyproject.toml file in project
#     e. Pipfile in project
#     f. Required python version: use_python(..., required = TRUE) if run
#        anytime at all prior in the session
#     g. RETICULATE_PYTHON_FALLBACK environment variable set
# - 2. FIND THE PYTHON OPTIONS
#     a. Suggested python version: use_python(..., required = FALSE)
#     b. Python virtual environments with name: r-reticulate
#     c. Python virtual environments with "required" module name
#     d. Python conda environments with name: r-reticulate (if installed
#        NORMALLY and not through homebrew)
#     e. Python conda environments with "required" module name (if installed
#        NORMALLY and not through homebrew)
#     f. NEW installation of miniconda in R (if enabled AND no other version of
#        Python has yet been flagged) with name: r-reticulate and after installing
#        the numpy package (if using renv, will install in renv and have
#        RETICULATE_MINICONDA_PYTHON_ENVPATH set to renv version, otherwise will
#        use base one)
#     g. NEW installation of miniconda in R (if enabled AND no other version of
#        Python has yet been flagged) with "required" module name and after
#        installing the numpy package (if using renv, will install in renv and have
#        RETICULATE_MINICONDA_PYTHON_ENVPATH set to renv version, otherwise will
#        use base one)
#     h. The default version of python3 on the system
#     i. Other common locations for python/conda on the system (this is where it
#        will find the homebrew installation)
#     j. Any other python environments that it detected earlier
# - 3. PICK THE VERSION ... cycle through a - j, and return if
#     a. Is NEW installation of miniconda IN r-reticulate environment OR
#     b. Python version >= 2.7, architecture OK, AND has numpy > 1.6
#     c. If everything else fails, choose python with architecture OK

reticulate::py_discover_config() # why is this one chosen ?

# You can fix the version of reticulate
# - Tools -> Global Options  -> Python Interpreter -> Select
# - Tools -> Project Options -> Python Interpreter -> Select
Sys.getenv("RETICULATE_PYTHON")
reticulate::py_discover_config() # why is this one chosen ?

# also change Renviron to .Renviron in project and restart
Sys.getenv("RETICULATE_PYTHON")
reticulate::py_discover_config() # why is this one chosen ?

# what if we have set the .Renviron AND fixed python in Project Options?

# Following all of the above is a headache! what should we do ... ?
# - Set Project Options for a project, and always use that version of python
#   in that project.
# - Use the use_python function WITH required to determine location of python at
#   the beginning of your session.
# - NEVER let R choose the heuristics! NEVER try to change anything after running
#   python!

## b. Vary Environment ---------------------------------------------------------

# probably best to install R version of miniconda
# (might need to increase download timeout time)
if(!file.exists(reticulate::miniconda_path())){
  options(timeout = 3200)
  reticulate::install_miniconda()
}

# environment name for SIOP workshop
siop_env <- "r-siop-nlp"

# creating required environments (in R version of miniconda)
reticulate::conda_create(envname = siop_env)  # reticulate::virtualenv_create()
                                              # if using venv

# activating FIRST environment (works)
reticulate::use_condaenv(condaenv = siop_env, # reticulate::use_virtualenv()
                         required = TRUE)     # if using venv
reticulate::py_discover_config()

# activating OTHER python (works)
reticulate::use_python("/usr/local/bin/python3")
reticulate::py_discover_config()

# using config and NOT discover_config
reticulate::py_config()

# trying to activate FIRST environment again
reticulate::use_condaenv(siop_env)                   # doesn't work
reticulate::use_condaenv(siop_env, required = FALSE) # doesn't work (hidden)
reticulate::py_discover_config()

# restart R and initialize environment
rstudioapi::restartSession()

# need to update renv to make this work (frustratingly doesn't always work)
siop_env <- "r-siop-nlp"
reticulate::use_condaenv(condaenv = siop_env,
                         required = TRUE)
reticulate::py_discover_config()

# if you are using renv, you can do the following, but be warned that it's very
# difficult to NOT use the renv python once you've done this!
# renv::use_python(python = file.path(reticulate::miniconda_path(), "envs", siop_env, "bin/python3"),
#                  type   = "conda")

# note: if using renv, it will force you to use renv/python/r-reticulate if
#       you DON'T select python using renv::use_python OR reticulate::use_python
# note: if using renv::use_python(), will put renv-python into RETICULATE_PYTHON
#       environment variable, and you cannot change environment. Need to use
#       renv::use_python and NOT reticulate::use_condaenv!
# see:  https://rstudio.github.io/renv/articles/python.html
# reticulate::conda_create("blah")
# reticulate::use_condaenv("blah", required = TRUE) # works ... NO warning
# reticulate::py_discover_config()                  # still OLD version

# 4.Using Python and R Code ====================================================

## a. Hello World --------------------------------------------------------------

# how can we write a simple "hello world" function using reticulate?

# way 1: use py_run_string (with a raw r string)
reticulate::py_run_string(r"(print("Hello World"))")

# way 2: import builtins module and use python methods/functions
builtins <- reticulate::import_builtins(convert = FALSE)
builtins$print("Hello World")

# way 3: use repl_python and use python interpreter
reticulate::repl_python() # print("Hello World")

## b. Accessing Objects --------------------------------------------------------

# you can use python in R pretty easily
reticulate::repl_python() # >>> y = 2

# and get things back into R!
reticulate::py$y

# and see/use R objects in python
x <- 14
reticulate::py_run_string("print(r.x)")

# note that py_run_string returns all of the python objects
reticulate::py_run_string("y = 2 + 2")$y # same as reticulate::py

# note: in python, use "object.method" to access methods/elements of objects
#       in R, use python_object$method to access corresponding methods

## c. Using Modules ------------------------------------------------------------

# you can install modules (doesn't default into current environment?)
reticulate::conda_install(packages = "numpy")      # bad (see "Package Plan")
reticulate::conda_install(envname  = siop_env,
                          packages = "numpy")      # good
reticulate:::condaenv_resolve()                    # default env to install

# reticulate has pretty strong opinions about where you should work! :/

# you can import packages and run things from them
np <- reticulate::import(module  = "numpy",
                         convert = FALSE)          # import package
X  <- np$array(c(2, 4, 1, 3))                      # run python code
X$mean()                                           # apply method(s)
Y  <- reticulate::py_to_r(X)
Y                                                  # is this an R object? restart R!

# restart R and initialize environment
rstudioapi::restartSession()                       # might need to FORCE a restart!
siop_env <- "r-siop-nlp"
reticulate::use_condaenv(siop_env)

# try again!!
builtins <- reticulate::import_builtins(convert = FALSE)
np       <- reticulate::import(module  = "numpy",
                               convert = FALSE)
X        <- np$array(c(2, 4, 1, 3))             
X$mean()                                  
Y        <- reticulate::py_to_r(X)
Y                                                  # NOW it works!

# note: if you use convert = TRUE, the output that you get in the py_to_r function
#       will be python output and not R output (it will do NOTHING)
# note: you cannot load a module multiple times in different ways!
# note: sometimes reticulate doesn't listen, then it's best to restart R (manually)!

# your variable is a pointer to a python object, so will update if python does
Z        <- X # original object
Z             # original x object
X$sort()      # sort in place
X             # is updated (in python)
Y             # is not updated (in R)
Z             # is updated (in python)

# you can chain operations
X$cumsum()$mean()

# how do we add two objects together in python
X2    <- X$copy()
Y2    <- X$copy()
X2 + Y2                                                 # doesn't work
np$add(X2, Y2)                                          # works
reticulate::py_run_string(code    = "Z2 = r.X2 + r.Y2", # works
                          convert = FALSE)$Z2

# so you can either
# - run python code directly using py_run_string (with r objects accessible in
#   the r python object)
# - use module/class methods in R using the "$", putting the R objects that point
#   to python objects in the function, and then converting the whole thing back
#   to R when needed

# but ... you generally want to use convert = FALSE and manually convert later!

# 6. Using R in Python Code ====================================================

# See 1-python_and_r_setup.py File

# WANT ABOUT 300 LINES OF CODE
