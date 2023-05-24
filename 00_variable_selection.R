# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(naniar)
library(doMC)
library(vip)

registerDoMC(cores = 4)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

########## load in data ######################################################
load("data/train.rda")
test <- read.csv("data/test.csv")

folds <- vfold_cv(train, v = 5, repeats = 3)

############## initial recipe for lasso/rf selection ############################
init_recipe <- recipe(y ~., data = train) %>%
  step_rm(id) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_impute_mean(all_predictors())

init_recipe %>% 
  prep() %>% 
  bake(new_data = NULL)

############## variable selection using lasso ###################################
lasso_mod <- logistic_reg(mode = "classification",
                        penalty = tune(),
                        mixture = 1) %>% 
  set_engine("glmnet")

lasso_params <- extract_parameter_set_dials(lasso_mod)
lasso_grid <- grid_regular(lasso_params, levels = 5)

lasso_workflow <- workflow() %>% 
  add_model(lasso_mod) %>% 
  add_recipe(init_recipe)

lasso_tune <- lasso_workflow %>% 
  tune_grid(resamples = folds,
            grid = lasso_grid)

lasso_wkflw_final <- lasso_workflow %>% 
  finalize_workflow(select_best(lasso_tune, metric = "roc_auc"))

lasso_fit <- fit(lasso_wkflw_final, data = train)

lasso_tidy <- lasso_fit %>% 
  tidy() %>%
  filter(estimate != 0 & estimate > 1e-05 | estimate < -1e-05)

View(lasso_tidy)

save(lasso_tidy, file = "data/lasso_variables.rda")

load("data/lasso_variables.rda")

############## variable selection using random forest #############################]

rf_mod <- rand_forest(mode = "classification",
                      mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity")

rf_params <- extract_parameter_set_dials(rf_mod) %>% 
  recipes::update(mtry = mtry(range = c(1,5)))

rf_grid <- grid_regular(rf_params, levels = 5)

rf_workflow <- workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(init_recipe)

rf_tune <- rf_workflow %>% 
  tune_grid(resamples = folds,
            grid = rf_grid)

rf_wkflw_final <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "roc_auc"))

rf_fit <- fit(rf_wkflw_final, data = train)

save(rf_fit, file = "data/rf_variables.rda")

load("data/rf_variables.rda")

rf_vip <- rf_fit %>% 
  extract_fit_parsnip %>% 
  vip()

save(rf_vip, file = "data/rf_vip.rda")

rf_vip

# x242, x382, x461,x526, x278, x249, x516, x549, x299, x041
lasso_vars <- lasso_tidy %>% 
  pull(term)
