#########################
### Imports and setup ###
#########################

library(tidyverse)
library(tidymodels)
library(embed) # for target encoding
#library(themis) # for smote

prep_df <- function(df) {
  # if('type' %in% colnames(df)) {
  #   df <- df %>% mutate(type = factor(type))
  # }
  return(df)
}

setup_train_recipe <- function(df, impute=F, encode=T){
  prelim_ft_eng <- recipe(type~., data=df) %>%
    step_rm(id) %>%
    step_mutate(color = factor(color))
  
  if(impute){
    prelim_ft_eng <- prelim_ft_eng %>% 
      #step_impute_median(all_numeric_predictors()
      # step_impute_linear(all_numeric_predictors(), impute_with =imp_vars(all_numeric_predictors()))
      # step_impute_knn(all_predictors(), impute_with =imp_vars(all_predictors()), neighbors=10)
      step_impute_bag(all_predictors(),impute_with=imp_vars(all_predictors()), trees=500)
  }

  # if(encode){
  #   prelim_ft_eng <- prelim_ft_eng %>%
  #     step_lencode_mixed(all_nominal_predictors(), outcome = vars(type))
  # }
  
  # prelim_ft_eng <- prelim_ft_eng %>%
  #   step_normalize(all_numeric_predictors())
  
  # Set up preprocessing
  prepped_recipe <- prep(prelim_ft_eng, new_data=df)
  
  return(prepped_recipe)
}