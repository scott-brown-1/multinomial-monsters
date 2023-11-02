#########################
### Imports and setup ###
#########################

# Import packages
library(tidyverse)
library(DataExplorer)
library(ggmosaic)

## Load data
ggg <- vroom::vroom('./data/train.csv')

###########################
####### Examine Data ######
###########################

## Exmaine dataframe and check data types; check shape
glimpse(ggg)
View(ggg)

## Fix dtypes: change applicable cols to factors
ggg['color'] <- factor(ggg$color)
ggg['type'] <- factor(ggg$type)

## Check categorical vs discrete vs continuous factors
plot_intro(ggg)

###########################
### Check Missing Values ##
###########################

# Count total missing values
sum(sum(is.na(ggg)))

# View missing values by feature
plot_missing(ggg)

###########################
## Visually examine data ##
###########################

ggplot(ggg, aes(x=color), stat=count) +
  geom_bar()

ggplot(data=ggg, aes(x=has_soul, y=type)) +
  geom_boxplot()

ggplot(data=ggg) + 
  geom_mosaic(aes(x=product(color), fill=type))

###########################
### Examine response var ##
###########################

ggplot(ggg, aes(x=type), stat=count) +
  geom_bar()