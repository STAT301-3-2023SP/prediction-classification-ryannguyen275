# Data Exploration
library(tidyverse)
library(tidymodels)

train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")

# best practice to split train data to prevent overfitting

set.seed(1234)
my_split <- initial_split(train, prop = 0.75, strata = y)

train_data <- training(my_split)
test_data <- testing(my_split)


################################################################
# Distribution of Y

ggplot(train, aes(y)) +
  geom_histogram()

# perfectly distributed

################################################################
# missingness

missing_lst <- list()

for(var in colnames(train)) {
  missing_lst[var] <- train %>% 
    select(any_of(var)) %>% 
    filter(is.na(!!sym(var))) %>% 
    summarize(num_missing = n())
}

missing_tbl <- enframe(unlist(missing_lst))

missing_tbl %>%
  mutate(pct = value/4034) %>% 
  arrange(desc(pct))


################################################################
# miscoded categorical variables

cat_lst <- list()

for(var in colnames(train)) {
  cat_lst[var] <- train %>% 
    select(any_of(var)) %>%
    # count unique values in variable
    summarize(unique = length(unique(!!sym(var))))
}

cat_tbl <- enframe(unlist(cat_lst)) %>% 
  filter(value <= 5)

train <- train %>% 
  mutate(x120 = as.factor(x120),
        x761 = as.factor(x761),
        y = as.factor(y))
save(train, file = "data/train.rda")

