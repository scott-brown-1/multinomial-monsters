#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(doParallel)

setwd('..')
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
  cl <- makePSOCKcluster(10)
  registerDoParallel(cl)
}

## Set up preprocessing
prepped_recipe <- setup_train_recipe(train, encode=T, 
                                     pca_threshold=0, scale_to_unit = 1)

## Bake recipe
bake(prepped_recipe, new_data=train)
bake(prepped_recipe, new_data=test)

#########################
## Fit Classifer Model ##
#########################

## Define model
mlp_model <- mlp(
  hidden_units = tune(),
  epochs = 75, #or 100 or 2507
  #activation="relu"
  ) %>%
  set_engine("nnet", verbose=0) %>%
  #set_engine("keras", verbose=0) %>%
  set_mode('classification')

## Define workflow
mlp_wf <- workflow() %>%
  add_recipe(prepped_recipe) %>%
  add_model(mlp_model)

## Grid of values to tune over
tuning_grid <- grid_regular(
  hidden_units(range=c(1, 30)),
  levels = 5)

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
cv_results <- mlp_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

## Find optimal tuning params
best_params <- cv_results %>%
  select_best("accuracy")

print(best_params)

hidden_unit_plot <- cv_results %>% 
  collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + 
  geom_line()

ggsave('hidden_unit_plot.png',plot=hidden_unit_plot)

## Fit workflow
final_wf <- mlp_wf %>%
  finalize_workflow(best_params) %>%
  fit(data = train)

## Predict new y
output <- predict(final_wf, new_data=test, type='class') %>%
  bind_cols(., test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

#LS: penalty, then mixture
vroom::vroom_write(output,'./outputs/mlp_predictions.csv',delim=',')

if(PARALLEL){
  stopCluster(cl)
}
