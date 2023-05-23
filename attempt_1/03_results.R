# Model Results
library(tidyverse)
library(tidymodels)
library(kableExtra)
library(vip)
library(doMC)
library(parallel)

tidymodels_prefer()

registerDoMC(cores = 4)

##### LOAD PACKAGES/DATA ##############################################

result_files <- list.files("attempt_2/results", "*.rda", full.names = TRUE)

for(i in result_files) {
  load(i)
}


####### PUT ALL GRIDS TG ############################################################
model_set <- as_workflow_set(
  "elastic_net" = en_tuned,
  "rand_forest" = rf_tuned, 
  "knn" = knn_tuned,
  "boosted_tree" = bt_tuned,
  "nn" = nn_tuned,
  "svm_poly" = svm_poly_tuned,
  "svm_radial" = svm_radial_tuned,
  "mars" = mars_tuned
)

## Table of results
model_results <- model_set %>% 
  group_by(wflow_id) %>% 
  mutate(best = map(result, show_best, metric = "rmse", n = 1)) %>% 
  select(best) %>% 
  unnest(cols = c(best))

save(model_results, file = "attempt_2/results/model_results.rda")

load("results/rf_tuned3.rda")

##### FINAL WORKFLOW ######################################################

rf_workflow <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))
 
##### FINAL FIT ######################################################
final_fit <- fit(rf_workflow, train)

predictions <- predict(final_fit, test) %>% 
  bind_cols(test %>% select(id)) %>% 
  rename(y = .pred)

write_csv(predictions, file = "submissions/18_submission.csv")

