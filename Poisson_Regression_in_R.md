### Learn what is Poisson Regression model, how it can be used to solve real-world problems and how to model Poisson Regression using R

**Prediction Analysis** is a part of data analysis, which uses many techniques from data mining and statistics to predict the future based on present data.

**Regression Analysis** is one of the supervised learning techniques used to analyze current data and predict future events. It uses set of available data called the *training data* that is used to create a regression model.This model can then be used to predict future events.

Regression analysis creates the relationship between the **response variable (the dependent variable)** and one or more **predictors** also called as the **independent variables**. It helps to understand how response variable changes if the predictor variable changes and therefore enables to predict future events.

**Note:** Response variable is the output/outcome variable also called as the *dependent variable*.Explanatory/Predictor variable is a variable that has an effect on the value of the response variable also called an *independent variable*. For example, a data set contains number of hours studied by students and their SAT test scores. In this example, number of hours studied is independent variable and SAT test scores is response variable.

The terms **Predictor**, **explanatory variable** are both used for independent variable. And the terms **Outcome varaible**, **Response variable** are both used for dependent variable.

There are many Regression Analysis techniques that can be used depending on the type of data variable. This tutorial explains the following aspects of Poisson Regression:

-   What is Poisson Regression and When to use it?
-   What is Poisson Distribution?
-   How Poisson Distribution differs from Normal Distribution?
-   Poisson Regression Model with GLMs
-   Modeling Poisson Regression for count data
-   Visualizing findings from model using jtools
-   Modeling Poisson Regression for rate data

### Poisson Regression model and use

Poisson Regression model is best used for events in which outcomes are counts. It tells which explanatory variables (X-values) have an effect on response variable (Y-values). For example, Poisson regression can be applied by a grocery store to predict the number of people in a line. Predictors/Explanatory variables may be number of items on discount offers or number of days remaining till some holiday.

To understand Poisson Regression,we must know what is meant by *count data*. Count data means discrete data such as the number of times an event will occur. It can also be presented as the *rate* over a time period or any other grouping.Following are some examples:

1.  Number of road accidents that occur during a month
2.  Number of goals scored by a player in a year
3.  Number of cases solved by a judge in a quarter

**Note:** The count for any event will be a positive number, that is 0,1,2,... ∞. It can never be a negative number. Poisson regression is used to model both, count data and rate data. In Poisson regression the response variable (Y) follows a *Poisson distribution*.

### Poisson Distribution

Poisson distribution models the probability of event *y*, that occurs randomly, by using the following formula called as **Probability Mass Function for Poisson Distribution**:

$$P(y)=\\frac{e^-\\mu.t(\\mu.t)^y}{y!}\\,\\,\\,\\,where\\,y=0,1,2...$$

Here:

*μ* is the average number of times an event may occur per unit of *exposure*. It is also called the **parameter** of Poisson distribution. In some text books *λ* may be used instead of *μ* . Exposure may be time, space, population size, distance, area but is often time that can be denoted by *t*. If exposure value is not given it is assumed to be equal to **1**.

**Note:** *Probability Mass Function (PMF)* of a discrete random variable (A variable whose possible values are discrete outcomes of a random event) lists the probabilities associated with each of the possible values.

Let's take a look at Poisson distribution plot for different values of *μ*. The R code is explained in the following steps:

Step 1: Create a vector of 6 colors Step 2: Create a list for the distribution that will have different values for *μ* Step 3: Create a vector of values for *μ* Step 4: Loop over the values from *μ* each with quantile range 0-20 and store in list Step 5: Plot the points using `plot()`

``` r
colors<-c("Red","Blue","Gold","Black","Pink","Green") #vector of colors

poisson.dist<-list() #declare a list to hold distribution values

a<-c(1,2,3,4,5,6) # A vector for values of u

for (i in 1:6) {
  poisson.dist[[i]]<-c(dpois(0:20,i)) 
  # Store distribution vector for each corresonding value of u
}
# plot each vector in the list using the colors vectors to represent each value for u
 plot(unlist(poisson.dist[1]),type = "o",xlab="y", ylab = "P(y)", col=colors[i])

for (i in 1:6) {
 lines(unlist(poisson.dist[i]),type = "o", col=colors[i])
  
}
 # Adds legend to the graph plotted
legend("topright", legend=a, inset= 0.08,  cex = 1.0, fill = colors, title = "Values of u")
```

![](Poisson_Regression_in_R_files/figure-markdown_github/Poisson%20Distribution-1.png)

`dpois(sequence,lambda)` is used to plot Probability Density Function (PDF) of Poisson distribution

**Information:** In probability theory, a probability density function (PDF), or density of a continuous random variable (A variable whose possible values are continuous outcomes of a random event), is a function that describes the relative likelihood for this random variable for a given value. In Statistics, random variable is simply a variable whose outcome is result of a random event.

### How Poisson Distribution differs from Normal Distribution?

Poisson Distribution is used to find out probability of events occurring at different time intervals which has lower bound 0 and no negative values are allowed. On the other hand, *Normal distribution* is a continuous distribution for a continuous variable with bounds - ∞ to ∞. Following are some key differences between Poisson and Normal Distribution:

<table style="width:21%;">
<colgroup>
<col width="11%" />
<col width="9%" />
</colgroup>
<thead>
<tr class="header">
<th align="right">Poisson Distribution</th>
<th align="left">Normal Distribution</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="right">Used for Count data or rate data (frequencies of events happening)</td>
<td align="left">Used for continuous variables</td>
</tr>
<tr class="even">
<td align="right">Skewed depending on Values of lambda. With lower mean it is highly skewed. All data pushed at <code>0</code> .As mean increases it looks like normal distribution but does not work like it</td>
<td align="left">Bell shaped curve. It is symmetric around the mean.</td>
</tr>
<tr class="odd">
<td align="right">In Poisson, Variance=Mean</td>
<td align="left">Variance and mean are different parameters. The mean, median and mode are equal</td>
</tr>
</tbody>
</table>

Normal Distribution can be generated in R as follows:

``` r
#create sequence -3 to +3 with .05 increment 
xseq<-seq(-3,3,.05) 
#generate a Probability density function
densities<-dnorm(xseq, 0,1) 
#plot the graph
plot(xseq, densities, col="blue",xlab="", ylab="Density", type="l",lwd=2) 
```

![](Poisson_Regression_in_R_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
#'col' changes thecolor of line
#'xlab' and 'ylab' are labels for x and y axis respectively
#'type' defines the type of plot. 'l' gives a line graph
# 'lwd' defines line width
```

In R,`dnorm(sequence,mean,std.dev)` is used to plot a Probability Density Function (PDF) of Normal Distribution.

To understand the Poisson distribution, consider the following problem:

**If there are twelve cars crossing a bridge per minute on average, what would be the probability of having seventeen or more cars crossing the bridge in a particular minute?**

Here, average number of cars crossing a bridge per minute is *μ*=12.

`ppois(q, u, lower.tail = TRUE)` is an R function that gives the probability that a random variable will be lower than or equal to a value.

As you have to find the probability of **having seventeen or more** cars, we will use `lower.trail=FALSE` and give q=16

``` r
ppois(16,12,lower.tail = FALSE)
```

    ## [1] 0.101291

``` r
#lower.tail=logical; if TRUE (default) then probabilities are P[X<=x], otherwise, P[X > x].
```

This means that there is **10.1%** probability of having 17 or more cars crossing the bridge in a particular minute.

### Poisson Regression Model and GLMs

Generalized Linear Models are models in which response variables follow a distribution other than the normal distribution. Whereas,in Linear regression model response variables follow normal distribution. This is because, Generalized Linear Models have response variables that are categorical such as Yes, No; or Group A, Group B and, therefore, do not range from -∞ to +∞. Hence, the relationship between response and predictor variables may not be linear. In GLM:

*y*<sub>*i*</sub> = *α* + *β*<sub>1</sub>*x*<sub>1</sub>*i* + *β*<sub>2</sub>*x*<sub>2</sub>*i* + ....+*β*<sub>*p*</sub>*x*<sub>*p*</sub>*i* + *e*<sub>*i*</sub>               *i* = 1, 2....*n*

The response variable *y*<sub>*i*</sub> is modeled by a *linear function of predictor variables* and some error term.

The Poisson Regression model is one of the *Generalized Linear Models (GLM)* that is used to model count data and contingency tables. The output *Y* (count) is a value that follows the Poisson distribution. It assumes the logarithm of *expected values (mean)* that can be modeled into a linear form by some unknown parameters.

**Information:** *In statistics, contingency tables are matrix of frequencies depending on multiple variables. [Click the given link to see a contingency table](https://i.stack.imgur.com/8YuSA.jpg)*

To transform the non-linear relationship to linear form, a **link function** is used which is the **log** for Poisson Regression. Therefore it is also called *log-linear model* .The general mathematical form of Poisson Regression model is:

*l**o**g*(*y*)=*α* + *β*<sub>1</sub>*x*<sub>1</sub> + *β*<sub>2</sub>*x*<sub>2</sub> + ....+*β*<sub>*p*</sub>*x*<sub>*p*</sub>
 Where,

-   *y*: Is the response variable
-   *α* and *β*: are numeric coefficients, *α* being the intercept, sometimes *α* also is represented by *β*<sub>0</sub>, it's the same
-   *x* is the predictor/explanatory variable

The coefficients are calculated using methods such as Maximum Likelihood Estimation(MLE) or [maximum quasi-likelihood](https://en.wikipedia.org/wiki/Quasi-likelihood)

Consider a equation simple with one predictor variables and one response variable:
*l**o**g*(*y*)=*α* + *β*(*x*)
 This is equivalent to,

*y* = *e*<sup>(*α* + *β*(*x*))</sup> = *e*<sup>*α*</sup> + *e*<sup>*β* \* *x*</sup>

**Note:** In Poisson Regression model predictor or explanatory variables can have a mixture of both numeric or categorical values.

One of the most important characteristics for Poisson distribution and Poisson Regression is called **Equi-dispersion** that the *mean and variance of the distribution are equal.*

**Note:** Variance measures the spread of the data. It is the **"average of the squared differences from the mean"**. Variance (Var) is equal to 0 if all values are identical. The greater the difference between the values the greater will be the variance. Mean is the average of values of a dataset. Average is the sum of the values divided by the number of values.

Let us say, that the mean (*μ*) is denoted by *E*(*X*)

*E*(*X*)=*μ*

For Poisson Regression, mean and variance are related as:

*v**a**r*(*X*)=*σ*<sup>2</sup>*E*(*X*)

Where *σ*<sup>2</sup> is the dispersion parameter. Since, *v**a**r*(*X*)=*E*(*X*)(variance=mean) must hold for the Poisson model to be completely fit, *σ*<sup>2</sup> must be equal to 1.

When variance is greater than mean, it is called **over-dispersion** and it is greater than 1. If it is less than 1 than it is known as **under-dispersion**.

### Poisson Regression modeling in R using glm()--Count Data

In R, `glm()` command is used to model Generalized Linear Models. Following is the general structure of the `glm()`:

                    glm(formula, family=familytype(link=""), data=,...)
                                              
                                              

In this tutorial, above three parameters are used in R. For further reading [here is the R documentation](https://www.rdocumentation.org/packages/stats/versions/3.4.3/topics/glm)

<table style="width:29%;">
<colgroup>
<col width="13%" />
<col width="15%" />
</colgroup>
<thead>
<tr class="header">
<th align="right">Parameter</th>
<th align="left">Description</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="right"><strong>formula</strong></td>
<td align="left">The formula is symbolic representation of how modeled is to fitted</td>
</tr>
<tr class="even">
<td align="right"><strong>family</strong></td>
<td align="left">Family tells choice of variance and link functions. There are 06 choices of family including Poisson and Logistic</td>
</tr>
<tr class="odd">
<td align="right"><strong>data</strong></td>
<td align="left">Data is the dataset to be used</td>
</tr>
</tbody>
</table>

\*\*<Tip:**> To learn about R formula [read this tutorial](https://www.datacamp.com/community/tutorials/r-formula-tutorial)

`glm()` provides six choices for family with their link functions as follows:

|            Family| Default Link Function                      |
|-----------------:|:-------------------------------------------|
|          binomial| (link = "logit")                           |
|          gaussian| (link = "identity")                        |
|             Gamma| (link = "inverse")                         |
|  inverse.gaussian| (link = $\\frac{1}{mu^2}$)                 |
|           poisson| (link = "log")                             |
|             quasi| (link = "identity", variance = "constant") |
|     quasibinomial| (link = "logit")                           |
|      quasipoisson| (link = "log")                             |

Let's model Poisson Regression on **The Number of Breaks in Yarn during Weaving** data that is part of the **datasets** package in R. You can install the package using `install.package("datasets")` and load the library with `library(datasets)`

``` r
#install.packages("datasets")
library(datasets) #include library datasets after installation
```

The above line of code will install the package. Then load the library using `library()` in R. and view the **warpbreaks** data. You can store in an object as below.

``` r
data<-warpbreaks
```

Let's view the data:

``` r
columns<-names(data) #Extract column names from dataframe
columns #show columns
```

    ## [1] "breaks"  "wool"    "tension"

#### Take a look at what's in your data

This data set gives the number of warp breaks per loom, where a loom corresponds to a fixed length of yarn. [Here is the description of dataset](http://stat.ethz.ch/R-manual/R-devel/library/datasets/html/warpbreaks.html)

It is a dataframe containing 54 observations on 3 variables.

|   Column| Type    | Description                   |
|--------:|:--------|:------------------------------|
|   breaks| numeric | The number of breaks          |
|     wool| factor  | The type of wool (A or B)     |
|  tension| factor  | The level of tension (L, M, H |

There are measurements on 9 looms for each of the six types of warp (AL, AM, AH, BL, BM, BH).

You can see the structure of data by `ls.str()` command:

``` r
ls.str(warpbreaks)
```

    ## breaks :  num [1:54] 26 30 54 25 70 52 51 26 67 18 ...
    ## tension :  Factor w/ 3 levels "L","M","H": 1 1 1 1 1 1 1 1 1 2 ...
    ## wool :  Factor w/ 2 levels "A","B": 1 1 1 1 1 1 1 1 1 1 ...

You can see that above structure shows you the type and levels present in the data.[Read this](https://www.statmethods.net/input/datatypes.html) to get more understanding of factors in R.

Now you will work with the `data` dataframe where `breaks` is the response variable and wool and tension are predictor variables. You can view the dependent variable `breaks` data continuity by creating a histogram:

``` r
hist(data$breaks) #generate histogram of new.data$whrswk
```

![](Poisson_Regression_in_R_files/figure-markdown_github/unnamed-chunk-7-1.png)

You can see that the data is not in the form a bell curve like in normal distribution. you can see the `mean()` and `var()` of the dependent variable as follows.

``` r
mean(data$breaks) #calculate mean
```

    ## [1] 28.14815

``` r
var(data$breaks) #calculate variance
```

    ## [1] 174.2041

You can see that the variance is much greater than the mean, so it suggests that you will have over-dispersion in the model.

Let's fit the Poisson model using the `glm()` command.

``` r
#model poisson regression usin glm()
poisson.model<-glm(breaks~wool+tension,data,family = poisson(link = "log"))
summary(poisson.model)
```

    ## 
    ## Call:
    ## glm(formula = breaks ~ wool + tension, family = poisson(link = "log"), 
    ##     data = data)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.6871  -1.6503  -0.4269   1.1902   4.2616  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  3.69196    0.04541  81.302  < 2e-16 ***
    ## woolB       -0.20599    0.05157  -3.994 6.49e-05 ***
    ## tensionM    -0.32132    0.06027  -5.332 9.73e-08 ***
    ## tensionH    -0.51849    0.06396  -8.107 5.21e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 297.37  on 53  degrees of freedom
    ## Residual deviance: 210.39  on 50  degrees of freedom
    ## AIC: 493.06
    ## 
    ## Number of Fisher Scoring iterations: 4

#### Interpreting the Poisson model

The next important thing is to interpret the model. The first column named `Estimate` are the coefficient values of *α* (intercept), *β*<sub>1</sub> and so on. Following is the interpretation for the parameter estimates:

-   *e**x**p*(*α*)= effect on the mean *μ*, when X = 0

-   *e**x**p*(*β*) = with every unit increase in X, the predictor variable has multiplicative effect of *e**x**p*(*β*) on the mean of Y, that is *μ*

-   If *β* = 0, then exp(*β*) = 1, and the expected count is *e**x**p*(*α*) and, Y and X are not related.

-   If *β* &gt; 0, then exp(*β*) &gt; 1, and the expected count is exp(*β*) times larger than when X = 0

-   If *β* &lt; 0, then exp(*β*) &lt; 1, and the expected count is exp(*β*) times smaller than when X = 0

If `family=poisson` is kept in `glm()` then, these parameters are calculated using [Maximum Likelihood Estimation MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

R treats categorical variables as dummy variables. Categorical variables, also called as indicator variables, are converted into dummy variables by assigning the levels in the variable some numeric representation.The general rule is:

**If there are *k* categories in a factor variable the output of glm() will have *k* − 1 categories with remaining 1 as the base category.**

You can see in summary that for wool 'A' has been made as base and is not shown in summary. Similarly, for tension 'L' has been made base category.

To see which explanatory variables have an effect on response variable you look at the *p* values. If the ***p is less than 0.05*** then, the variable has an effect on the response variable. In summary above, all p values are less than 0.05, hence, all have significant effect on breaks. Notice how R output used `***` at the end of each variable. *The Number of stars is depicting significance.*

Before starting to interpret results, check if the model has over-dispersion or under-dispersion. If you see that the *Residual Deviance* is greater than the degrees of freedom, then over-dispersion exists. This means **That the estimates are correct, but the standard errors (standard deviation are wrong and unaccounted by the model**.

**Note:** The Null deviance shows how well the response variable is predicted by a model that includes only the intercept (grand mean) whereas residual with the inclusion of independent variables. Above, you can see that addition of 3 (53-50 =3) independent variables decreased the deviance to 210.39 from 297.37. Greater difference in values mean a bad fit.

So, to have a more correct standard error you can use *quasi-poisson* model:

``` r
poisson.model2<-glm(breaks~wool+tension,data=data,family = quasipoisson(link = "log"))
summary(poisson.model)
```

    ## 
    ## Call:
    ## glm(formula = breaks ~ wool + tension, family = poisson(link = "log"), 
    ##     data = data)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.6871  -1.6503  -0.4269   1.1902   4.2616  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  3.69196    0.04541  81.302  < 2e-16 ***
    ## woolB       -0.20599    0.05157  -3.994 6.49e-05 ***
    ## tensionM    -0.32132    0.06027  -5.332 9.73e-08 ***
    ## tensionH    -0.51849    0.06396  -8.107 5.21e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 297.37  on 53  degrees of freedom
    ## Residual deviance: 210.39  on 50  degrees of freedom
    ## AIC: 493.06
    ## 
    ## Number of Fisher Scoring iterations: 4

#### Comparing both models:

``` r
#install.packages("arm")
# load library arm that contains the function se.coef()
library(arm) 
```

    ## Warning: package 'arm' was built under R version 3.4.4

    ## Loading required package: MASS

    ## Loading required package: Matrix

    ## Loading required package: lme4

    ## Warning: package 'lme4' was built under R version 3.4.4

    ## 
    ## arm (Version 1.10-1, built: 2018-4-12)

    ## Working directory is D:/Documents/R

``` r
#extract co-effcients from first model using 'coef()'
coef1=coef(poisson.model) 
#extract co-effcients from second model
coef2=coef(poisson.model2) 
#extract standard errors from first model using 'se.coef()'
se.coef1=se.coef(poisson.model) 
 #extract standard errors from second model
se.coef2=se.coef(poisson.model2)
#use 'cbind()' to combine values one dataframe
models.both<-cbind(coef1,se.coef1,coef2,se.coef2,exponent=exp(coef1)) 
# show dataframe
models.both 
```

    ##                  coef1   se.coef1      coef2   se.coef2   exponent
    ## (Intercept)  3.6919631 0.04541069  3.6919631 0.09374352 40.1235380
    ## woolB       -0.2059884 0.05157117 -0.2059884 0.10646089  0.8138425
    ## tensionM    -0.3213204 0.06026580 -0.3213204 0.12440965  0.7251908
    ## tensionH    -0.5184885 0.06395944 -0.5184885 0.13203462  0.5954198

In above output, you can see the coefficient are same but Standard errors are different.

Keeping these points in mind, let us see estimate for *wool*. Its value is **-0.2059884**, and exponent of **-0.2059884** is **0.8138425**

``` r
1-0.8138425
```

    ## [1] 0.1861575

This shows that, if we change wool type from A to B then there will be *decrease* in breaks *0.8138425* times than the intercept, because estimate -0.2059884 is negative. Another way of saying this is **If we change wool type from A to B then number of breaks would fall by 18.6% if we keep all other variables same**

#### Predicting from the model

`predict(model, data, type)` is used to predict from the model. Once the model is made, you can use it to predict outcomes using new dataframes containing data other than the training data.

``` r
#make a datframe with new data
newdata = data.frame(wool="B",tension="M")
#use 'predict() to run model on new data
predict(poisson.model2,newdata = newdata,type="response")
```

    ##        1 
    ## 23.68056

So, number of breaks may be **24** with wool **type=B and tension=M**.

### Visualizing findings from the model using `jtools`

When you are sharing your analysis with others, tables are often not the best way to grab people's attention. Plots and graphs help people grasp your findings quicker.

Here you will use an awesome R package called [jtools](https://cran.r-project.org/web/packages/jtools/jtools.pdf) that includes tools for summarizing and visualizing your regression models. It can be used for all regression models. Let's use `jtools` for our Poisson model `poisson.model2`.

``` r
#Install the package jtools if not already installed
#install.packages("jtools")
#you may be asked to install 'broom' and 'ggstance' package as well
#install.packages("broom")
#install.packages("ggstance")
```

`jtools` provides `plot_summs()` and `plot_coefs()` to visualized summary of the model and also allows you to compare different models with `ggplot2`.

``` r
#Include jtools library
library(jtools)
```

    ## Warning: package 'jtools' was built under R version 3.4.4

    ## 
    ## Attaching package: 'jtools'

    ## The following object is masked from 'package:arm':
    ## 
    ##     standardize

``` r
#plot regression coefficients for poisson.model2
plot_summs(poisson.model2,scale=TRUE,exp=TRUE)
```

    ## Note: Pseudo-R2 for quasibinomial/quasipoisson families is calculated
    ## by refitting the fitted and null models as binomial/poisson.

![](Poisson_Regression_in_R_files/figure-markdown_github/unnamed-chunk-15-1.png)

``` r
#plot regression coefficients for poisson.model2 and poisson.model
plot_summs(poisson.model,poisson.model2,scale=TRUE,exp=TRUE)
```

    ## Note: Pseudo-R2 for quasibinomial/quasipoisson families is calculated
    ## by refitting the fitted and null models as binomial/poisson.

![](Poisson_Regression_in_R_files/figure-markdown_github/unnamed-chunk-15-2.png) In above code, the `plot_summs(poisson.model2,scale=TRUE,exp=TRUE)` plots the second model using quasi-poisson family in glm.

-   The first argument in `plot_summs()` is the regression model to be used, it may be one or more than one.
-   `scale` helps with the problem of differing scales of the variables.
-   `exp` is set to TRUE because for Poisson regression we are more likely to be interested in exponential values of estimates than linear.

You can find more details on jtools and `plot_summs()` [here](https://www.jtools.jacob-long.com/reference/plot_summs.html)

You can also visualize the interaction between predictor variables. `jtools` provide different functions for different types of variables. For example if all the variables are categorical you should use `cat_plot()` to better understand interactions among them. For continuous variables, `interact_plot()` is used.

In the *warpbreaks* data we have categorical predictor variables, so let us use `cat_plot()` to visualize the interaction between them.

``` r
# using cat_plot. Pass poisson.model2 and we want to see effect of wool type so we set pred=wool
cat_plot(poisson.model2, pred = wool, modx = tension)
```

![](Poisson_Regression_in_R_files/figure-markdown_github/unnamed-chunk-16-1.png)

``` r
#argument 1=regression model
#pred=The categorical variable that will appear on x-axis
#modx=A categorical moderator variable. Moderator variable that has an effect in combination to pred on outcome
```

Similarly, you can use it for tension:

``` r
# using cat_plot. Pass poisson.model2 and we want to see effect of tension type so we set pred=tension
cat_plot(poisson.model2, pred = tension, modx=wool)
```

![](Poisson_Regression_in_R_files/figure-markdown_github/unnamed-chunk-17-1.png) You can see it gives us how the three different categories of tension L,M and H for each wool type A and B effects breaks.

Next, you can also define the type of plot in `cat_plot()` using `geom` parameter. This parameter enhances the interpretation of plot. You can use it as below:

``` r
# using cat_plot. Pass poisson.model2 and we want to see effect of tension type so we set pred=tension
cat_plot(poisson.model2, pred = tension, modx=wool,geom ="line")
```

![](Poisson_Regression_in_R_files/figure-markdown_github/unnamed-chunk-18-1.png)

If you want to include observations also in your plot add `plot.points=TRUE`:

``` r
# using cat_plot. Pass poisson.model2 and we want to see effect of tension type so we set pred=tension
cat_plot(poisson.model2, pred = tension, modx=wool,geom = "line", plot.points = TRUE)
```

![](Poisson_Regression_in_R_files/figure-markdown_github/unnamed-chunk-19-1.png) There are lots of other decoration options such as line style, color set you can use according to your requirements. You can find it [Here](https://www.jtools.jacob-long.com/articles/categorical.html).

### Poisson Regression modelling in R using glm()--Rate Data

So far this in this tutorial, you have modeled count data, you can model rate data that is predicting the number of counts over a period of time or grouping. Formula for modelling rate data is given by:

*l**o**g*(*X*/*n*)=*β*<sub>0</sub> + ∑<sub>*i*</sub>*β*<sub>*i*</sub>*X*<sub>*i*</sub>

This is equivalent to: (applying log formula)

*l**o**g*(*X*)−*l**o**g*(*n*)=*β*<sub>0</sub> + ∑<sub>*i*</sub>*β*<sub>*i*</sub>*X*<sub>*i*</sub>

*l**o**g*(*X*)=*l**o**g*(*n*)+*β*<sub>0</sub> + ∑<sub>*i*</sub>*β*<sub>*i*</sub>*X*<sub>*i*</sub>

Thus, it can be modeled by including the **log(n)** term with coefficient of 1. This is called an **offset**. This offset is modelled with **offset()** in R.

Let's use another dataset in **ISwR** package called **eba1977** to model Poisson Regression Model for rate data.

``` r
#install.packages("ISwR")
library(ISwR)
```

    ## Warning: package 'ISwR' was built under R version 3.4.4

``` r
data(eba1977)
cancer.data=eba1977
cancer.data[1:10,]
```

    ##          city   age  pop cases
    ## 1  Fredericia 40-54 3059    11
    ## 2     Horsens 40-54 2879    13
    ## 3     Kolding 40-54 3142     4
    ## 4       Vejle 40-54 2520     5
    ## 5  Fredericia 55-59  800    11
    ## 6     Horsens 55-59 1083     6
    ## 7     Kolding 55-59 1050     8
    ## 8       Vejle 55-59  878     7
    ## 9  Fredericia 60-64  710    11
    ## 10    Horsens 60-64  923    15

``` r
#Description
#Lung cancer incidence in four Danish cities 1968-1971

#Description:
#     This data set contains counts of incident lung cancer cases and
#     population size in four neighbouring Danish cities by age group.

#Format:
#     A data frame with 24 observations on the following 4 variables:
#     city a factor with levels Fredericia, Horsens, Kolding, and Vejle.
#     age a factor with levels 40-54, 55-59, 60-64, 65-69,70-74, and 75+.
#     pop a numeric vector, number of inhabitants.
 #    cases a numeric vector, number of lung cancer cases.
```

To model rate data which is **X/n** where *X* is event to happen and *n* is the grouping. In this example, **X=cases** that is event will be a case of cancer and **n=pop** that is population is the grouping.

As in formula above rate data is accounted by *l**o**g*(*n*) and in this data *n* is population, so you will find log of population first. You can model for *cases/population* as follows:

``` r
#find out log(n) of each value in 'pop' column. It is the third column
logpop=log(cancer.data[,3])
#add the log values to the dataframe using 'cbind()'
new.cancer.data=cbind(cancer.data,logpop)
#display new dataframe
new.cancer.data
```

    ##          city   age  pop cases   logpop
    ## 1  Fredericia 40-54 3059    11 8.025843
    ## 2     Horsens 40-54 2879    13 7.965198
    ## 3     Kolding 40-54 3142     4 8.052615
    ## 4       Vejle 40-54 2520     5 7.832014
    ## 5  Fredericia 55-59  800    11 6.684612
    ## 6     Horsens 55-59 1083     6 6.987490
    ## 7     Kolding 55-59 1050     8 6.956545
    ## 8       Vejle 55-59  878     7 6.777647
    ## 9  Fredericia 60-64  710    11 6.565265
    ## 10    Horsens 60-64  923    15 6.827629
    ## 11    Kolding 60-64  895     7 6.796824
    ## 12      Vejle 60-64  839    10 6.732211
    ## 13 Fredericia 65-69  581    10 6.364751
    ## 14    Horsens 65-69  834    10 6.726233
    ## 15    Kolding 65-69  702    11 6.553933
    ## 16      Vejle 65-69  631    14 6.447306
    ## 17 Fredericia 70-74  509    11 6.232448
    ## 18    Horsens 70-74  634    12 6.452049
    ## 19    Kolding 70-74  535     9 6.282267
    ## 20      Vejle 70-74  539     8 6.289716
    ## 21 Fredericia   75+  605    10 6.405228
    ## 22    Horsens   75+  782     2 6.661855
    ## 23    Kolding   75+  659    12 6.490724
    ## 24      Vejle   75+  619     7 6.428105

``` r
#model rate data using offset()
poisson.model.rate <- glm(cases ~ city + age+offset(logpop), family = poisson(link = "log"), data =cancer.data)
#display summary
summary(poisson.model.rate)
```

    ## 
    ## Call:
    ## glm(formula = cases ~ city + age + offset(logpop), family = poisson(link = "log"), 
    ##     data = cancer.data)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -2.63573  -0.67296  -0.03436   0.37258   1.85267  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  -5.6321     0.2003 -28.125  < 2e-16 ***
    ## cityHorsens  -0.3301     0.1815  -1.818   0.0690 .  
    ## cityKolding  -0.3715     0.1878  -1.978   0.0479 *  
    ## cityVejle    -0.2723     0.1879  -1.450   0.1472    
    ## age55-59      1.1010     0.2483   4.434 9.23e-06 ***
    ## age60-64      1.5186     0.2316   6.556 5.53e-11 ***
    ## age65-69      1.7677     0.2294   7.704 1.31e-14 ***
    ## age70-74      1.8569     0.2353   7.891 3.00e-15 ***
    ## age75+        1.4197     0.2503   5.672 1.41e-08 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 129.908  on 23  degrees of freedom
    ## Residual deviance:  23.447  on 15  degrees of freedom
    ## AIC: 137.84
    ## 
    ## Number of Fisher Scoring iterations: 5

In this dataset you can see that Residual deviance is near to degrees of freedom, the dispersion parameter is **1.5 (23.447/15)** which is small, so the model is a good fit.

`fittted(model)` is used to return values fitted by the model. It returns outcomes using the training data on which the model is built.

``` r
fitted(poisson.model.rate)
```

    ##         1         2         3         4         5         6         7 
    ## 10.954812  7.411803  7.760169  6.873215  8.615485  8.384458  7.798635 
    ##         8         9        10        11        12        13        14 
    ##  7.201421 11.609373 10.849479 10.092831 10.448316 12.187276 12.576313 
    ##        15        16        17        18        19        20        21 
    ## 10.155638 10.080773 11.672630 10.451942  8.461440  9.413988  8.960422 
    ##        22        23        24 
    ##  8.326004  6.731286  6.982287

You can predict the number of cases per 1000 population for a new data set, using the `predict()` function as follows:

``` r
#create a test dataframe containing new values of variables
test.data=data.frame(city="Kolding",age="40-54", pop=1000, logpop=log(1000)) 
#predict outcomes (responses) using 'predict()' 
predicted.value<-predict(poisson.model.rate, test.data, type = "response") 
#show predicted value
predicted.value
```

    ##        1 
    ## 2.469818

So, **For the city of Kolding, people in the age group between 40-54 there is a possibility of 3 cases of lung cancer per 1000 of the population**.

You can use quasi-poisson to get more correct standard errors in rate data as in count data.

That's the end of the tutorial
------------------------------

Poisson regression models have great significance in econometric and real world predictions. In this tutorial, you learned what is Poisson Distribution, Generalized Linear Models, and Poisson Regression model. You also learned how to implement Poisson Regression Model for both count and rate data in R using `glm()` and how to fit the data to the model to predict for a new dataset. You went through how to get more accurate standard errors in glm() using *'quasipoisson'*. Hope this tutorial increased your statistical knowledge and hands-on R experience! Happy Learning.

### References

1.  <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Poisson.html>
2.  <https://www.theanalysisfactor.com/generalized-linear-models-in-r-part-6-poisson-regression-count-variables/>
3.  <https://stats.idre.ucla.edu/r/dae/poisson-regression/>
4.  <https://onlinecourses.science.psu.edu/stat504/node/169/>
5.  <https://onlinecourses.science.psu.edu/stat504/node/165/>
