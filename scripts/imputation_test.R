#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)

#setwd('..')
source('./scripts/ggg_analysis.R')
PARALLEL <- T

#########################
####### Load Data #######
#########################

## Load data
train_na <- prep_df(vroom::vroom('./data/train_with_na.csv'))
train <- prep_df(vroom::vroom('./data/train.csv'))
test <- prep_df(vroom::vroom('./data/test.csv'))

#########################
## Feature Engineering ## 
#########################

set.seed(843)

## parallel tune grid
if(PARALLEL){
  cl <- makePSOCKcluster(8)
  registerDoParallel(cl)
}
## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, impute = T)

## Bake recipe
bake(prepped_recipe, new_data=train) #train
bake(prepped_recipe, new_data=test)

## Calc imputation RMSE
train <- bake(prepped_recipe, new_data=train)
train_imputed <- bake(prepped_recipe, new_data=train_na)

calc_rmse <- function(){
  errs <- c()
  
  for(col in c('bone_length','rotting_flesh','hair_length')){
    e <- rmse_vec(
      train[col][is.na(train_na[col])],
      train_imputed[col][is.na(train_na[col])])
  
    errs <- append(errs, e)
  
    print(col)
    print(paste0(col,': ', e))
  }
  
  cat(mean(errs))
}

calc_rmse()

# Median: 0.15
# lm: 0.14
# KNN: 0.13
# Bag: 0.11

