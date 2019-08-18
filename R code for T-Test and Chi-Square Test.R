##############################################################################
#  T-TEST DEMO
##############################################################################

library(dplyr)
library(tidyr)
data(tips, package = "reshape2")
glimpse(tips)
## Observations: 244
## Variables: 7
## $ total_bill <dbl> 16.99, 10.34, 21.01, 23.68, 24.59, 25.29, 8.77, 26....
## $ tip        <dbl> 1.01, 1.66, 3.50, 3.31, 3.61, 4.71, 2.00, 3.12, 1.9...
## $ sex        <fctr> Female, Male, Male, Male, Female, Male, Male, Male...
## $ smoker     <fctr> No, No, No, No, No, No, No, No, No, No, No, No, No...
## $ day        <fctr> Sun, Sun, Sun, Sun, Sun, Sun, Sun, Sun, Sun, Sun, ...
## $ time       <fctr> Dinner, Dinner, Dinner, Dinner, Dinner, Dinner, Di...
## $ size       <int> 2, 3, 3, 2, 4, 4, 2, 4, 2, 2, 2, 4, 2, 4, 2, 2, 3, ...

##########################  METHOD 1  ##########################
tips %>%
    select(tip, total_bill, sex) %>%
    gather(key = variable, value = value, -sex) %>%
    group_by(sex, variable) %>%
    summarise(value = list(value)) %>%
    spread(sex, value) %>%
    group_by(variable) %>%
    mutate(p_value = t.test(unlist(Female), unlist(Male))$p.value,
           t_value = t.test(unlist(Female), unlist(Male))$statistic)
## Source: local data frame [2 x 5]
## Groups: variable [2]
##
##     variable     Female        Male    p_value   t_value
##        <chr>     <list>      <list>      <dbl>     <dbl>
## 1        tip <dbl [87]> <dbl [157]> 0.13780684 -1.489536
## 2 total_bill <dbl [87]> <dbl [157]> 0.01857339 -2.373398

##########################  GO THROUGH METHOD 1  ##########################
tips %>%
    select(tip, total_bill, sex) %>% head
##    tip total_bill    sex
## 1 1.01      16.99 Female
## 2 1.66      10.34   Male
## 3 3.50      21.01   Male
## 4 3.31      23.68   Male
## 5 3.61      24.59 Female
## 6 4.71      25.29   Male
tips %>%
    select(tip, total_bill, sex) %>%
    gather(key = variable, value = value, -sex) %>% head
##      sex variable value
## 1 Female      tip  1.01
## 2   Male      tip  1.66
## 3   Male      tip  3.50
## 4   Male      tip  3.31
## 5 Female      tip  3.61
## 6   Male      tip  4.71

#we "melt" the data frame down, so that all numeric variables are put in one column (underneath each other).

tips %>%
    select(tip, total_bill, sex) %>%
    gather(key = variable, value = value, -sex) %>%
    group_by(sex, variable) %>%
    summarise(value = list(value))
## Source: local data frame [4 x 3]
## Groups: sex [?]
##
##      sex   variable       value
##   <fctr>      <chr>      <list>
## 1 Female        tip  <dbl [87]>
## 2 Female total_bill  <dbl [87]>
## 3   Male        tip <dbl [157]>
## 4   Male total_bill <dbl [157]>

# Now it get's interesting. We put all the values per group (e.g., male-tip or female-total_bill.) in one cell. Yes, that's right. In each cell of column value there is now a list (a bunch) of values. That's what is called a "list-column". We will now use this list column for the t-test.

tips %>%
    select(tip, total_bill, sex) %>%
    gather(key = variable, value = value, -sex) %>%
    group_by(sex, variable) %>%
    summarise(value = list(value)) %>%
    spread(sex, value) %>%
    group_by(variable)
## Source: local data frame [2 x 3]
## Groups: variable [2]
##
##     variable     Female        Male
## *      <chr>     <list>      <list>
## 1        tip <dbl [87]> <dbl [157]>
## 2 total_bill <dbl [87]> <dbl [157]>

# But before we do the t-Test, we "spread" the data frame. That is, we convert from "long" to "wide" format. Next, we group for variable. That means in practice, that the following t-test will be applied to each member of this group (ie., each variable, here tip and total_bill).
# And now the t-Test:

    tips %>%
    select(tip, total_bill, sex) %>%
    gather(key = variable, value = value, -sex) %>%
    group_by(sex, variable) %>%
    summarise(value = list(value)) %>%
    spread(sex, value) %>%
    group_by(variable) %>%
    mutate(p_value = t.test(unlist(Female), unlist(Male))$p.value,
           t_value = t.test(unlist(Female), unlist(Male))$statistic)
## Source: local data frame [2 x 5]
## Groups: variable [2]
##
##     variable     Female        Male    p_value   t_value
##        <chr>     <list>      <list>      <dbl>     <dbl>
## 1        tip <dbl [87]> <dbl [157]> 0.13780684 -1.489536
## 2 total_bill <dbl [87]> <dbl [157]> 0.01857339 -2.373398

##########################  METHOD 2  ##########################
#You can have it very simple

t.test(tip ~ sex, data = tips)$p.value
## [1] 0.1378068
t.test(tip ~ sex, data = tips)$estimate
t.test(total_bill ~ sex, data = tips)$p.value
## [1] 0.01857339

##############################################################################
#  CHI-SQUARE TEST DEMO
##############################################################################

library(gplots)
library(graphics)
library(vcd)
library(corrplot)
# Import the data
file_path <- "http://www.sthda.com/sthda/RDoc/data/housetasks.txt"
housetasks <- read.delim(file_path, row.names = 1)

# transform for antother data into the format we want
if (FALSE) {
    dt = data.table::fread("./test.csv")
    dt2 <- as.table(as.matrix(dt[, c(2,3)]))
    row.names(dt2) = dt$Experience
}
# here, the Laundry, etc. are just index
# head(housetasks)
#           Wife Alternating Husband Jointly
#Laundry     156          14       2       4
#Main_meal   124          20       5       4
#Dinner       77          11       7      13
#Breakfeast   82          36      15       7
#Tidying      53          11       1      57
#Dishes       32          24       4      53

# 1. convert the data as a table
dt <- as.table(as.matrix(housetasks))
# 2. Graph
balloonplot(t(dt), main ="housetasks", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE)
mosaicplot(dt, shade = TRUE, las=2,
           main = "housetasks")
# plot just a subset of the table
assoc(head(dt, 5), shade = TRUE, las=3)

###############
# Chi-Square Test to see if rows are independent of columns
###############
chisq <- chisq.test(housetasks)
chisq
# The small p-value indicates two columns are dependent
#Pearson's Chi-squared test
#X-squared = 1944.5, df = 36, p-value < 2.2e-16
# Observed counts
chisq$observed
# Expected counts
round(chisq$expected,2)
# get Pearson residuals
round(chisq$residuals, 3)
# plot the Pearson residuals
corrplot(chisq$residuals, is.cor = FALSE)
# Contibution in percentage (%) to CHi-square score
contrib <- 100*chisq$residuals^2/chisq$statistic
round(contrib, 3)
# Visualize the contribution
corrplot(contrib, is.cor = FALSE)
# printing the p-value
chisq$p.value
# printing the mean
chisq$estimate
