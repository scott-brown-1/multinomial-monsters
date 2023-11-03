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
prepped_recipe <- setup_train_recipe(train, encode=T, pca_threshold=0)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
####### Fit Model #######
#########################

# NOTE: WORK IN PROGRESS SCRIPT

## Define model
rand_forest_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 2
  ) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## Define workflow
rand_forest_wf <- workflow(prepped_recipe) %>%
  add_model(rand_forest_model)

## Grid of values to tune over
tuning_grid <- grid_regular(
  mtry(range=c(1,5)),
  min_n(),
  levels = 5)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
cv_results <- rand_forest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find optimal tuning params
best_params <- cv_results %>%
  select_best("roc_auc")

print(best_params)

# Fit workflow
final_wf <- rand_forest_wf %>%
  finalize_workflow(best_params) %>%
  fit(data = train)

final_wf <- rand_forest_wf %>%
  fit(data = train)

## Predict new y
output <- predict(final_wf, new_data=test, type='class') %>%
  bind_cols(., test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

vroom::vroom_write(output,'./outputs/svm_preds.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}