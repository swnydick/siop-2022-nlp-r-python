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

# 2. Required Packages =========================================================

# if you don't have the latest version of each package, please install them
packageVersion("reticulate") # install.packages("reticulate")
packageVersion("tensorflow") # install.packages("tensorflow")
packageVersion("keras")      # install.packages("keras")

# we will always use the "pkg::fun()" syntax for reticulate for clarity.
median(c(1, 2, 3, 4))        # if a package is loaded, you can use its name
stats::median(c(1, 2, 3, 4)) # use the "::" syntax to not fully load the package

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
#        the numpy package
#     g. NEW installation of miniconda in R (if enabled AND no other version of
#        Python has yet been flagged) with "required" module name and after
#        installing the numpy package
#     h. The default version of python3 on the system
#     i. Other common locations for python/conda on the system (this is where it
#        will find the homebrew installation)
#     j. Any other python environments that it detected earlier
# - 3. PICK THE VERSION ... cycle through a - j, and return if
#     a. Is NEW installation of miniconda IN r-reticulate environment OR
#     b. Python version >= 2.7, architecture OK, AND has numpy > 1.6
#     c. If everything else fails, choose python with architecture OK
reticulate::py_discover_config()

# You can fix the version of reticulate
# - Tools -> Global Options  -> Python Interpreter -> Select
# - Tools -> Project Options -> Python Interpreter -> Select
Sys.getenv("RETICULATE_PYTHON")

# also change Renviron to .Renviron in project and restart
Sys.getenv("RETICULATE_PYTHON")
reticulate::py_discover_config()

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
reticulate::conda_create(envname = siop_env)

# activating FIRST environment (works)
reticulate::use_condaenv(siop_env)
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
reticulate::use_condaenv(siop_env)
reticulate::py_discover_config()

# 4. Using Python in R Code ====================================================

# you can install modules (doesn't default into current environment?)
reticulate::conda_install(packages = "numpy")      # bad
reticulate::conda_install(envname  = siop_env,
                          packages = "numpy")      # good
reticulate:::condaenv_resolve()                    # default env to install

# reticulate has pretty strong opinions about where you should work! :/

# you can use python in R pretty easily
reticulate::repl_python() # >>> y = 2

# and get things back into R!
reticulate::py$y

# and see/use R objects in python
x <- 14
reticulate::py_run_string("print(r.x)")

# note that py_run_string returns all of the python objects
reticulate::py_run_string("y = 2 + 2")$r$x

# note: in python, use "object.method" to access methods/elements of objects
#       in R, use python_object$method to access corresponding methods

# you can import packages and run things from them
np <- reticulate::import(module  = "numpy",
                         convert = FALSE) # import package
X  <- np$array(c(2, 4, 1, 3))             # run python code
X$mean()                                  # apply method(s)
y  <- reticulate::py_to_r(X)

# note: if you use convert = TRUE, the output that you get in the py_to_r function
#       will be python output and not R output (it will do NOTHING)
# note: you cannot load a module multiple times in different ways! (rule of reticulate)

# your variable is a pointer to a python object, so will update if python does
X$sort() # sort in place
X        # is updated (in python)
y        # is not updated (in R)

# you can chain operations
X$cumsum()$mean()

# how do we add two objects together in python
X2    <- X$copy()
Y2    <- X$copy()
Z2    <- reticulate::py_run_string(code    = "Z2 = r.X2 + r.Y2",
                                   convert = FALSE)$Z2

# 5. Using R in Python Code ====================================================

# See 1-python_and_r_setup.py File

# WANT ABOUT 300 LINES OF CODE
