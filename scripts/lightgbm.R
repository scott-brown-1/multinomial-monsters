#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)
library(bonsai)

source('./scripts/ggg_analysis.R')
PARALLEL <- F

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
prepped_recipe <- setup_train_recipe(train, encode=F, pca_threshold=0)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Regression Model #
#########################

boost_model <- boost_tree(
  trees = tune(), #tune(), #200
  tree_depth = tune(),#tune(), #5,
  learn_rate = 0.1,
  mtry = 3,#tune(), #3,
  min_n = 25,#tune(), #20,
  loss_reduction = 0 #tune(), #0
) %>% 
  set_engine("lightgbm") %>% 
  set_mode("classification")

## Define workflow
# Transform response to get different cutoff
boost_wf <- workflow(prepped_recipe) %>%
  add_model(boost_model)

# Grid of values to tune over
tuning_grid <- grid_regular(
 trees(),
 tree_depth(),
 #learn_rate(),
 #mtry(range=c(3,ncol(train))),
 #min_n(),
 #loss_reduction(),
 levels = 3)

## Split data for CV
folds <- vfold_cv(train, v = 3, repeats=1)

## Run the CV
cv_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
           metrics=metric_set(roc_auc))

# Find optimal tuning params
best_params <- cv_results %>% 
  select_best("roc_auc")

print(best_params)

# Fit workflow
final_wf <- boost_wf %>%
  finalize_workflow(best_params) %>%
  fit(data = train)

## Predict new y
output <- predict(final_wf, new_data=test, type='class') %>%
  bind_cols(., test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

#LS: penalty, then mixture
vroom::vroom_write(output,'./outputs/light_gbm_preds.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
