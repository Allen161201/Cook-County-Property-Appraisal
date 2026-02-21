# ============================================
# Cook County Property Assessment - FINAL
# Expected runtime: 14 minutes
## ============================================


## --------------------------------------------------------------------------------
# Load packages and data
library(tidyverse)
library(randomForest)

historic <- read_csv("historic_property_data.csv")
predict_data <- read_csv("predict_property_data.csv")

cat("Historic data:", nrow(historic), "rows,", ncol(historic), "columns\n")
cat("Predict data:", nrow(predict_data), "rows,", ncol(predict_data), "columns\n")


## --------------------------------------------------------------------------------
# Variable Selection Process
# I started with 60+ variables but full RF was too slow (estimated 44 hours)
# Used 10k sample to test variable importance and iteratively selected features

# Check available columns
cat("Total columns in historic:", ncol(historic), "\n")

# Based on importance testing and domain knowledge, I selected:
# Numeric (9): meta_certified_est_bldg, meta_certified_est_land, char_bldg_sf, 
#              char_fbath, char_rooms, char_age, char_hd_sf, econ_midincome, econ_tax_rate
# Categorical (1): char_type_resd
# Will add: target encoding (2), interactions (2), derived features (1)

# Verify these columns exist
selected_vars <- c("meta_certified_est_bldg", "meta_certified_est_land", 
                   "char_bldg_sf", "char_fbath", "char_rooms", "char_age", 
                   "char_hd_sf", "econ_midincome", "econ_tax_rate", "char_type_resd",
                   "meta_nbhd", "geo_school_elem_district")

cat("Variables found:", sum(selected_vars %in% names(historic)), "of", length(selected_vars), "\n")
cat("Missing:", setdiff(selected_vars, names(historic)), "\n")

## --------------------------------------------------------------------------------
# Target Encoding for high-cardinality categorical variables
# meta_nbhd has 845 levels - too many for one-hot encoding
# Convert to neighborhood mean price instead

# Neighborhood mean price
nbhd_stats <- historic %>%
  group_by(meta_nbhd) %>%
  summarise(nbhd_mean_price = mean(sale_price, na.rm = TRUE))

historic <- left_join(historic, nbhd_stats, by = "meta_nbhd")
predict_data <- left_join(predict_data, nbhd_stats, by = "meta_nbhd")

# School district mean price
school_stats <- historic %>%
  group_by(geo_school_elem_district) %>%
  summarise(school_mean_price = mean(sale_price, na.rm = TRUE))

historic <- left_join(historic, school_stats, by = "geo_school_elem_district")
predict_data <- left_join(predict_data, school_stats, by = "geo_school_elem_district")

# Fill missing with overall mean
overall_mean <- mean(historic$sale_price, na.rm = TRUE)
predict_data$nbhd_mean_price[is.na(predict_data$nbhd_mean_price)] <- overall_mean
predict_data$school_mean_price[is.na(predict_data$school_mean_price)] <- overall_mean

cat("Target encoding complete\n")
cat("Unique neighborhoods:", n_distinct(historic$meta_nbhd), "\n")
cat("Unique school districts:", n_distinct(historic$geo_school_elem_district), "\n")

## --------------------------------------------------------------------------------
# Preprocessing: handle missing values, create derived features and interactions

preprocess <- function(df, is_train = TRUE) {
  df <- df %>%
    mutate(
      # Handle missing values
      econ_midincome = ifelse(is.na(econ_midincome), 
                              median(econ_midincome, na.rm = TRUE), econ_midincome),
      char_fbath = ifelse(is.na(char_fbath), 1, char_fbath),
      char_rooms = ifelse(is.na(char_rooms), 
                          median(char_rooms, na.rm = TRUE), char_rooms),
      char_age = ifelse(is.na(char_age), 
                        median(char_age, na.rm = TRUE), char_age),
      char_hd_sf = ifelse(is.na(char_hd_sf), 0, char_hd_sf),
      char_type_resd = factor(ifelse(is.na(char_type_resd), "Missing", char_type_resd)),
      
      # Derived features
      total_est = meta_certified_est_bldg + meta_certified_est_land,
      
      # Interaction terms
      interaction_sf_fbath = char_bldg_sf * char_fbath,
      interaction_sf_rooms = char_bldg_sf * char_rooms
    )
  
  if (is_train) {
    df %>% select(sale_price, 
                  meta_certified_est_bldg, meta_certified_est_land, total_est,
                  char_bldg_sf, char_fbath, char_rooms, char_age, char_hd_sf,
                  econ_midincome, econ_tax_rate,
                  char_type_resd,
                  nbhd_mean_price, school_mean_price,
                  interaction_sf_fbath, interaction_sf_rooms)
  } else {
    df %>% select(pid,
                  meta_certified_est_bldg, meta_certified_est_land, total_est,
                  char_bldg_sf, char_fbath, char_rooms, char_age, char_hd_sf,
                  econ_midincome, econ_tax_rate,
                  char_type_resd,
                  nbhd_mean_price, school_mean_price,
                  interaction_sf_fbath, interaction_sf_rooms)
  }
}

train_final <- preprocess(historic, is_train = TRUE)
test_final <- preprocess(predict_data, is_train = FALSE)

# Verify no NAs in key columns
cat("Missing values in training data:\n")
print(colSums(is.na(train_final)))


## --------------------------------------------------------------------------------
# Train Random Forest model
# 200 trees, 15 variables (9 numeric + 1 categorical + 2 target encoding + 2 interaction + 1 derived)

set.seed(123)
train_x <- train_final %>% select(-sale_price)
train_y <- train_final$sale_price

cat("Training RF with", ncol(train_x), "variables on", nrow(train_x), "rows\n")
cat("Started at:", format(Sys.time(), "%H:%M:%S"), "\n")

rf_model <- randomForest(
  x = train_x,
  y = train_y,
  ntree = 200,
  importance = FALSE
)

cat("Finished at:", format(Sys.time(), "%H:%M:%S"), "\n")
cat("OOB RMSE:", round(sqrt(rf_model$mse[200])), "\n")

## --------------------------------------------------------------------------------
# Generate predictions
test_x <- test_final %>% select(-pid)
predictions <- predict(rf_model, newdata = test_x)

# Cap predictions at max training price
max_price <- max(historic$sale_price)
cat("Max training price:", max_price, "\n")
cat("Predictions above max:", sum(predictions > max_price), "\n")

predictions_capped <- pmin(predictions, max_price)

# Create output dataframe
output <- data.frame(
  pid = test_final$pid,
  assessed_value = predictions_capped
)

# Verify format
cat("\nOutput rows:", nrow(output), "\n")
cat("PID range:", min(output$pid), "to", max(output$pid), "\n")
cat("Any missing values:", sum(is.na(output$assessed_value)), "\n")
cat("Any negative values:", sum(output$assessed_value < 0), "\n")

# Summary statistics for conclusion
cat("\nAssessed Value Summary:\n")
print(summary(output$assessed_value))

# Export
write.csv(output, "assessed_value.csv", row.names = FALSE)
cat("\nExported to assessed_value.csv\n")

## --------------------------------------------------------------------------------
# Error Analysis: RMSE by price segment
# This reveals why luxury homes are hard to predict

# Get OOB predictions from RF
oob_predictions <- rf_model$predicted

# Create analysis dataframe
error_analysis <- data.frame(
  actual = train_final$sale_price,
  predicted = oob_predictions,
  error = train_final$sale_price - oob_predictions
)

# Define price segments
error_analysis <- error_analysis %>%
  mutate(
    price_segment = case_when(
      actual < 600000 ~ "< $600K",
      actual < 800000 ~ "$600K - $800K",
      actual < 1000000 ~ "$800K - $1M",
      actual < 1500000 ~ "$1M - $1.5M",
      TRUE ~ "> $1.5M"
    ),
    price_segment = factor(price_segment, levels = c("< $600K", "$600K - $800K", 
                                                      "$800K - $1M", "$1M - $1.5M", "> $1.5M"))
  )

# Calculate RMSE by segment
segment_rmse <- error_analysis %>%
  group_by(price_segment) %>%
  summarise(
    count = n(),
    pct = round(n() / nrow(error_analysis) * 100, 1),
    rmse = round(sqrt(mean(error^2))),
    .groups = "drop"
  )

#Table 1: RMSE by Price Segment
print(segment_rmse)

## --------------------------------------------------------------------------------
# Appendix: Visualizations

# Figure 1: RMSE by Price Segment
p1 <- ggplot(segment_rmse, aes(x = price_segment, y = rmse)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = scales::comma(rmse)), vjust = -0.5, size = 3) +
  labs(title = "Figure 1: RMSE by Price Segment",
       x = "Price Segment", y = "RMSE ($)") +
  scale_y_continuous(labels = scales::comma) +
  theme_minimal()
print(p1)

# Figure 2: Distribution of Assessed Values
p2 <- ggplot(output, aes(x = assessed_value)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Figure 2: Distribution of Assessed Property Values",
       x = "Assessed Value ($)", y = "Count") +
  scale_x_continuous(labels = scales::comma) +
  theme_minimal()
print(p2)

# Figure 3: Actual vs Predicted (training data)
p3 <- ggplot(error_analysis, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.1) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Figure 3: Actual vs Predicted Sale Price (OOB)",
       x = "Actual Sale Price ($)", y = "Predicted Sale Price ($)") +
  scale_x_continuous(labels = scales::comma, limits = c(0, 3000000)) +
  scale_y_continuous(labels = scales::comma, limits = c(0, 3000000)) +
  theme_minimal()
print(p3)

