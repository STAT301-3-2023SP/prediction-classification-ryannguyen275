# Boosted Tree Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer()

load("attempt_1/setups/setup1.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
bt_model <- boost_tree(mode = "classification",
                       mtry = tune(),
                       min_n = tune(),
                       learn_rate = tune()) %>% 
  set_engine("xgboost", importance = "impurity")


bt_param <- extract_parameter_set_dials(bt_model) %>% 
  recipes::update(mtry = mtry(range = c(1,10)))

bt_grid <- grid_regular(bt_param, levels = 5)

bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(recipe1)

##### TUNE GRID ########################################################

bt_tuned <- tune_grid(bt_workflow,
                      resamples = folds,
                      grid = bt_grid,
                      verbose = TRUE,
                      control = control_grid(save_pred = TRUE, 
                                             save_workflow = TRUE,
                                             verbose = TRUE,
                                             parallel_over = "everything"))


save(bt_tuned, bt_workflow, file = "attempt_1/results/bt_tuned.rda")