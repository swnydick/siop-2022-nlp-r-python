
<!-- README.md is generated from README.Rmd. Please edit that file -->

## Workshop Setup

This workshop folder uses the `renv` package. This package helps
standardize and install appropriate `R` packages required by this
repository and *nothing else*. That way, your package setup (at least)
will be similar to the package setup used when these materials were
being constructed. That said, using `renv` does not ensure that your
entire system will be setup correctly (see the [renv
introduction](https://rstudio.github.io/renv/articles/renv.html) for
more information), only that you will have the required `R` packages and
versions.

To install and load the appropriate packages:

1.  Make sure that you are connected to the internet.
2.  Make sure that you have installed
    [RStudio](https://www.rstudio.com/products/rstudio/download/#download).
3.  Open the
    [siop-2022-interactive-shiny.Rproj](siop-2022-interactive-shiny.Rproj)
    in RStudio.
4.  Run the following code to restore the library:

``` r
renv::restore(prompt = FALSE)
```

If you do not wish to use `renv`, feel free to open up the `R` files in
RStudio and install all of the required packages manually. You can find
the list of required packages by opening the [renv.lock](renv.lock)
file.
