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
load("data/lasso_variables.rda")

train <- initial_split(train, prop = 0.75, strata = y)
train <- training(train)
test <- testing(train)

folds <- vfold_cv(train, v = 5, repeats = 3, strata = y)

########## set up recipes #####################################################

# RECIPE 1: Lasso Kitchen Sink
lasso_var <- lasso_tidy %>% 
  pull(term)

train_lasso <- train %>% 
  select(any_of(lasso_var), y)

recipe1 <- recipe(y ~ ., data = train_lasso) %>%
  step_nzv(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_impute_mean(all_numeric_predictors())

recipe1 %>% 
  prep() %>% 
  bake(new_data = NULL)

save(recipe1, folds, file = "attempt_1/setups/setup1.rda")
