library(tidyverse)
library(caret)
library(randomForest)
library(e1071)

hotels <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv")
head(hotels)
str(hotels)
summary(hotels)

ggplot(hotels, aes(x = is_canceled)) + geom_bar(fill = "tomato", color = "black") + theme_minimal() + ggtitle("Booking Cancellation Frequency")

ggplot(hotels, aes(x = hotel)) + 
  geom_bar(fill = "steelblue") + 
  theme_minimal() + 
  labs(title = "Distribution of Hotel Types", x = "Hotel", y = "Count")
pairs(~lead_time+adr+stays_in_week_nights+total_of_special_requests, data = hotels)

numerical_features <- hotels %>% select_if(is.numeric)
correlation_matrix <- cor(numerical_features, use = "complete.obs")
corrplot::corrplot(correlation_matrix, method = "circle")

cancellation_rate <- mean(hotels$is_canceled)
print(paste("Cancellation Rate: ", cancellation_rate))

ggplot(hotels, aes(x = hotel, fill = as.factor(is_canceled))) + 
  geom_bar(position = "fill") + 
  scale_fill_manual(values = c("red", "green"), 
                    labels = c("Not Canceled", "Canceled")) + 
  labs(y = "Proportion", title = "Cancellation by Hotel Type")


ggplot(hotels |>
         filter(adr < 4000) |> 
         mutate(total_nights = stays_in_weekend_nights+stays_in_week_nights),
       aes(x = adr, y = total_nights)) +
  geom_point(alpha=0.1)

hotels <- hotels |>
  filter(adr < 4000) |> 
  mutate(total_nights = stays_in_weekend_nights+stays_in_week_nights)

ggplot(hotels,
       aes(x = adr, y = total_nights)) +
  geom_bin2d(binwidth=c(10,1)) +
  geom_smooth()
hotels <- hotels |>
  select(-reservation_status, -reservation_status_date) |> 
  mutate(kids = case_when(
    children + babies > 0 ~ "kids",
    TRUE ~ "none"
  ))


hotels <- hotels |> 
  select(-babies, -children)


hotels <- hotels |> 
  mutate(parking = case_when(
    required_car_parking_spaces > 0 ~ "parking",
    TRUE ~ "none"
  )) |> 
  select(-required_car_parking_spaces)

library("GGally")

ggpairs(hotels |> select(kids, adr, parking, total_of_special_requests),
        aes(color = kids))

hotels.bycountry <- hotels |> 
  group_by(country) |> 
  summarise(total = n(),
            cancellations = sum(is_canceled),
            pct.cancelled = cancellations/total*100)
ggplot(hotels.bycountry |> arrange(desc(total)) |> head(10),
       aes(x = country, y = pct.cancelled)) +
  geom_col()
hotels.par <- hotels |>
  select(hotel, is_canceled, kids, meal, customer_type) |>
  group_by(hotel, is_canceled, kids, meal, customer_type) |>
  summarize(value = n())

install.packages("ggforce")
library("ggforce")

ggplot(hotels.par |> gather_set_data(x = c(1, 3:5)),
       aes(x = x, id = id, split = y, value = value)) +
  geom_parallel_sets(aes(fill = as.factor(is_canceled)),
                     axis.width = 0.1,
                     alpha = 0.66) + 
  geom_parallel_sets_axes(axis.width = 0.15, fill = "lightgrey") + 
  geom_parallel_sets_labels(angle = 0) +
  coord_flip()
hotels2 <- hotels |> 
  select(-country, -reserved_room_type, -assigned_room_type, -agent, -company,
         -stays_in_weekend_nights, -stays_in_week_nights)

fit.lr <- glm(as.factor(is_canceled) ~ ., binomial, hotels2)
summary(fit.lr)
pred.lr <- predict(fit.lr, hotels2, type = "response")
ggplot(data.frame(x = pred.lr), aes(x = x)) + geom_histogram()

conf.mat <- table(`true cancel` = hotels2$is_canceled, `predict cancel` = pred.lr > 0.5)
conf.mat


library(caret)
hotels2$is_canceled <- factor(hotels2$is_canceled, levels = c(0, 1), labels = c("NotCanceled", "Canceled"))

cv_control <- trainControl(
  method = "cv",              
  number = 10,                
  classProbs = TRUE,          
  summaryFunction = twoClassSummary  
)

set.seed(123)  
cv_model <- train(
  is_canceled ~ .,
  data = hotels2,
  method = "glm",
  family = "binomial",
  trControl = cv_control,
  metric = "ROC"
)

print(cv_model)
install.packages("naivebayes")
library(naivebayes)

cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary 
)

set.seed(123)
nb_model <- train(
  is_canceled ~ .,
  data = hotels2,
  method = "naive_bayes",
  trControl = cv_control,
  metric = "ROC"
)

# Summarize the Naive Bayes model
print(nb_model)

roc_comparison <- resamples(list(LogisticRegression = cv_model, NaiveBayes = nb_model))
summary(roc_comparison)
