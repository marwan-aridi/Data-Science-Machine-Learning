# Load necessary libraries
library(readr)
library(dplyr)

library(modelr)
library(broom)

library(caret)

library(randomForest)

# Load the data
data <- read_csv("Downloads/Cancer Surgeries CA Hospitals 2013-2022.csv")

# Filter out 'Statewide' entries and drop 'LONGITUDE' and 'LATITUDE' columns
data_filtered <- data %>%
  filter(hospital != "Statewide") %>%
  select(-LONGITUDE, -LATITUDE)

# Compute variance of '# of Cases' grouped by 'Hospital', 'Surgery', and 'County'
hospital_variance <- data_filtered %>%
  group_by(hospital) %>%
  summarise(hospital_effect = var(`# of Cases`, na.rm = TRUE))

surgery_variance <- data_filtered %>%
  group_by(Surgery) %>%
  summarise(surgery_effect = var(`# of Cases`, na.rm = TRUE))

county_variance <- data_filtered %>%
  group_by(County) %>%
  summarise(county_effect = var(`# of Cases`, na.rm = TRUE))

# Join these variances back to the main data
data_model <- data_filtered %>%
  left_join(hospital_variance, by = "hospital") %>%
  left_join(surgery_variance, by = "Surgery") %>%
  left_join(county_variance, by = "County")

# Remove rows with NA values
data_model <- na.omit(data_model)

# Fit the linear regression model using the aggregated variables
model <- lm(`# of Cases` ~ Year + hospital_effect + surgery_effect + county_effect, data = data_model)

# Summarize the model to see coefficients
summary_model <- summary(model)

# Print model summary
print(summary_model)

# Show the residual plots
par(mfrow = c(2,2))
plot(model)

# Fit the Random Forest model using the aggregated variables
rf.model <- randomForest(`# of Cases` ~ Year + hospital_effect + surgery_effect + county_effect, data = data_model, ntree = 500)

# Print model summary
print(rf.model)

# Plot variable importance
importance <- randomForest::importance(rf.model)
print(importance)

randomForest::varImpPlot(rf.model)

#---------------------------------------------------------#
set.seed(10)

# Set up 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Fit the linear model using caret for cross-validation
linear_model_cv <- train(
  `# of Cases` ~ Year + hospital_effect + surgery_effect + county_effect, 
  data = data_model,
  method = "lm",
  trControl = ctrl
)



#---------------------------------------------------------#


library(ranger)

# Set up k-fold cross-validation manually
k <- 10
folds <- cut(seq(1, nrow(data_model)), breaks = k, labels = FALSE)

# Perform 10-fold cross-validation
results <- data.frame(MSE = rep(NA, k))


for(i in 1:k){
  # Segment data into training and testing
  testIndexes <- which(folds == i, arr.ind = TRUE)
  testData <- data_model[testIndexes, ]
  trainData <- data_model[-testIndexes, ]

  # Fit Random Forest model on training data
  rf_train <- ranger(
    formula         = `# of Cases` ~ Year + hospital_effect + surgery_effect + county_effect, 
    data            = trainData,
    num.trees       = 500,
    importance      = 'impurity'
  )

  # Predict on test data
  predictions <- predict(rf_train, data = testData)$predictions

  # Calculate MSE
  results$MSE[i] <- mean((testData$`# of Cases` - predictions)^2)
}

# Calculate the average MSE
mean_mse <- mean(results$MSE)

# Print the average MSE
print(mean_mse)

# Extract MSE for linear regression
linear_mse <- linear_model_cv$results$RMSE^2


# Print MSE values
print(linear_mse)

