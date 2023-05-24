# Model Results
library(tidyverse)
library(tidymodels)

tidymodels_prefer()

##### LOAD PACKAGES/DATA ##############################################

result_files <- list.files("attempt_2/results", "*.rda", full.names = TRUE)

for(i in result_files) {
  load(i)
}

load("attempt_1/results/svm_poly_tuned.rda")
load("data/train.rda")
test <- read_csv("data/test.csv")

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
  mutate(best = map(result, show_best, metric = "roc_auc", n = 1)) %>% 
  select(best) %>% 
  unnest(cols = c(best))

save(model_results, file = "attempt_1/results/model_results.rda")

load("attempt_1/results/model_results.rda")

##### FINAL WORKFLOW ######################################################

svm_poly_workflow <- svm_poly_workflow %>% 
  finalize_workflow(select_best(svm_poly_tuned, metric = "roc_auc"))
 
##### FINAL FIT ######################################################
final_fit <- fit(svm_poly_workflow, train)

predictions <- predict(final_fit, test) %>% 
  bind_cols(test %>% select(id)) %>% 
  rename(y = .pred_class)

write_csv(predictions, file = "submissions/submission_3.csv")

