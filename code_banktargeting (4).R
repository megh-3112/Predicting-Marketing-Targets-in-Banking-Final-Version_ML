#Importing of train and test data
train_data <- read.csv2("D:/Adv stat/Marketing Targets/train.csv", 
                        stringsAsFactors = FALSE)
test_data <- read.csv2("D:/Adv stat/Marketing Targets/test.csv", 
                        stringsAsFactors = FALSE)

#Data Exploration
# Check structure
str(train_data)
# View first few rows
head(train_data)
# Summary of all columns
summary(train_data)
# Total missing values per column
colSums(is.na(train_data))

# Total missing values in the whole dataset
sum(is.na(train_data))
# Example for some columns
sapply(train_data[c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y")], unique)

#test data
colSums(is.na(test_data))
str(test_data)

#Visualization
library(ggplot2)
library(dplyr)

# Proportional distribution of Age
ggplot(train_data, aes(x = age)) +
  geom_histogram(aes(y = after_stat(count) / sum(after_stat(count))), bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Proportional Distribution of Age", x = "Age", y = "Proportion")

# Proportional distribution of Job roles
ggplot(train_data, aes(x = job)) +
  geom_bar(aes(y = after_stat(count) / sum(after_stat(count))), fill = "lightgreen") +
  labs(title = "Proportional Distribution of Job Roles", x = "Job", y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Proportional distribution of Education levels
ggplot(train_data, aes(x = education)) +
  geom_bar(aes(y = after_stat(count) / sum(after_stat(count))), fill = "salmon") +
  labs(title = "Proportional Distribution of Education Levels", x = "Education", y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Proportional distribution of Target variable (Subscribed y)
ggplot(train_data, aes(x = y)) +
  geom_bar(aes(y = after_stat(count) / sum(after_stat(count))), fill = "orchid") +
  labs(title = "Proportional Distribution of Target Variable (y) in train data", x = "Subscribed (y)", y = "Proportion")
ggplot(test_data, aes(x = y)) +
  geom_bar(aes(y = after_stat(count) / sum(after_stat(count))), fill = "orchid") +
  labs(title = "Proportional Distribution of Target Variable (y) in test data", x = "Subscribed (y)", y = "Proportion")

ggplot(train_data, aes(x = job, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Proportion of Subscription by Job", x = "Job", y = "Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(train_data, aes(x = education, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Subscription Rate by Education Level", x = "Education", y = "Proportion")
ggplot(train_data, aes(x = housing, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Subscription Rate by Housing Loan", x = "Housing", y = "Proportion")

ggplot(train_data, aes(x = loan, fill = y)) +
  geom_bar(position = "fill") +
  labs(title = "Subscription Rate by Personal Loan", x = "Loan", y = "Proportion")
# Select only numeric variables
numeric_cols <- sapply(train_data, is.numeric)
train_numeric <- train_data[, numeric_cols]
library(reshape2)
corr_matrix <- round(cor(train_numeric), 2) 
melted_corr <- melt(corr_matrix)
ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), color = "black", size = 4) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 10, hjust = 1)) +
  labs(title = "Correlation Heatmap of Numeric Variables",
       x = "", y = "")

#VIF analysis
library(car)
# Subset numeric features only
numeric_data <- train_data[, sapply(train_data, is.numeric)]
# Remove target variable if included
numeric_data$y <- NULL
calculate_vif <- function(df) {
  # Keep only numeric variables
  df_numeric <- df[, sapply(df, is.numeric)]
  
  vif_values <- sapply(names(df_numeric), function(var) {
    predictors <- setdiff(names(df_numeric), var)
    formula <- as.formula(paste(var, "~", paste(predictors, collapse = " + ")))
    model <- lm(formula, data = df_numeric)
    r_squared <- summary(model)$r.squared
    vif <- 1 / (1 - r_squared)
    return(vif)
  })
  
  # Return as sorted data frame
  vif_df <- data.frame(Variable = names(vif_values), VIF = round(vif_values, 2))
  vif_df <- vif_df[order(-vif_df$VIF), ]
  return(vif_df)
}
# Calculate VIFs
vif_results <- calculate_vif(train_data)
print(vif_results)

#Processing
# Get character columns
cat_cols <- sapply(train_data, is.character)

# Convert to factors in both train and test
train_data[cat_cols] <- lapply(train_data[cat_cols], as.factor)
test_data[cat_cols] <- lapply(test_data[cat_cols], as.factor)

# Get numeric columns
num_cols <- sapply(train_data, is.numeric)
# Scale train numeric data
train_scaled <- scale(train_data[, num_cols])
train_data[, num_cols] <- train_scaled

# Use same scaling parameters for test data
test_scaled <- scale(test_data[, num_cols],
                     center = attr(train_scaled, "scaled:center"),
                     scale = attr(train_scaled, "scaled:scale"))
test_data[, num_cols] <- test_scaled

#Model Building
library(caret)
library(e1071)
library(MASS)
library(pROC)
library(ROCR)
library(ggplot2)
library(randomForest)

# Ensure target is factor with "yes" as positive class
train_data$y <- factor(train_data$y, levels = c("no", "yes"))
test_data$y <- factor(test_data$y, levels = c("no", "yes"))

x_train <- train_data[, setdiff(names(train_data), "y")]
y_train <- train_data$y

x_test <- test_data[, setdiff(names(test_data), "y")]
y_test <- test_data$y

evaluate_model <- function(model_name, y_true, y_prob, y_pred) {
  cat("\n---", model_name, "---\n")
  
  # Confusion Matrix
  cm <- confusionMatrix(y_pred, y_true, positive = "yes")
  print(cm)
  
  # ROC Curve
  roc_obj <- roc(response = y_true, predictor = y_prob, levels = rev(levels(y_true)))
  auc_val <- auc(roc_obj)
  cat("AUC:", auc_val, "\n")
  
  plot(roc_obj, main = paste("ROC Curve -", model_name), col = "blue")
  
  # Precision, Recall, F1
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- cm$byClass["F1"]
  acc <- cm$overall["Accuracy"]
  
  cat(sprintf("Precision: %.3f | Recall: %.3f | F1 Score: %.3f | Accuracy: %.3f\n",
              precision, recall, f1, acc))
}

evaluate_model_dual <- function(model_name, y_true_test, y_prob_test, y_pred_test,
                                y_true_train, y_prob_train, y_pred_train) {
  cat("\n---", model_name, " (TEST) ---\n")
  cm_test <- confusionMatrix(y_pred_test, y_true_test, positive = "yes")
  print(cm_test)
  roc_test <- roc(response = y_true_test, predictor = y_prob_test, levels = rev(levels(y_true_test)))
  cat("AUC (Test):", auc(roc_test), "\n")
  plot(roc_test, main = paste("ROC Curve -", model_name, "(Test)"), col = "blue")
  cat(sprintf("Precision: %.3f | Recall: %.3f | F1 Score: %.3f | Accuracy: %.3f\n",
              cm_test$byClass["Precision"], cm_test$byClass["Recall"], cm_test$byClass["F1"],
              cm_test$overall["Accuracy"]))
  
  cat("\n---", model_name, " (TRAIN) ---\n")
  cm_train <- confusionMatrix(y_pred_train, y_true_train, positive = "yes")
  print(cm_train)
  roc_train <- roc(response = y_true_train, predictor = y_prob_train, levels = rev(levels(y_true_train)))
  cat("AUC (Train):", auc(roc_train), "\n")
  plot(roc_train, main = paste("ROC Curve -", model_name, "(Train)"), col = "green")
  cat(sprintf("Precision: %.3f | Recall: %.3f | F1 Score: %.3f | Accuracy: %.3f\n",
              cm_train$byClass["Precision"], cm_train$byClass["Recall"], cm_train$byClass["F1"],
              cm_train$overall["Accuracy"]))
}

#Logistic Regression
logit_model <- glm(y ~ ., data = train_data, family = binomial)

# Test predictions
logit_probs_test <- predict(logit_model, newdata = test_data, type = "response")
logit_preds_test <- ifelse(logit_probs_test > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))

# Train predictions
logit_probs_train <- predict(logit_model, newdata = train_data, type = "response")
logit_preds_train <- ifelse(logit_probs_train > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))

evaluate_model_dual("Logistic Regression", y_test, logit_probs_test, logit_preds_test,
                    y_train, logit_probs_train, logit_preds_train)


#Naive Bayes
nb_model <- naiveBayes(y ~ ., data = train_data)

nb_probs_test <- predict(nb_model, newdata = test_data, type = "raw")[, "yes"]
nb_preds_test <- predict(nb_model, newdata = test_data)

nb_probs_train <- predict(nb_model, newdata = train_data, type = "raw")[, "yes"]
nb_preds_train <- predict(nb_model, newdata = train_data)

evaluate_model_dual("Naive Bayes", y_test, nb_probs_test, nb_preds_test,
                    y_train, nb_probs_train, nb_preds_train)


# LDA (Linear Discriminant Analysis)
lda_model <- lda(y ~ ., data = train_data)

lda_pred_test <- predict(lda_model, newdata = test_data)
lda_probs_test <- lda_pred_test$posterior[, "yes"]
lda_preds_test <- lda_pred_test$class

lda_pred_train <- predict(lda_model, newdata = train_data)
lda_probs_train <- lda_pred_train$posterior[, "yes"]
lda_preds_train <- lda_pred_train$class

evaluate_model_dual("LDA", y_test, lda_probs_test, lda_preds_test,
                    y_train, lda_probs_train, lda_preds_train)

#QDA (Quadratic Discriminant Analysis)
qda_model <- qda(y ~ ., data = train_data)

qda_pred_test <- predict(qda_model, newdata = test_data)
qda_probs_test <- qda_pred_test$posterior[, "yes"]
qda_preds_test <- qda_pred_test$class

qda_pred_train <- predict(qda_model, newdata = train_data)
qda_probs_train <- qda_pred_train$posterior[, "yes"]
qda_preds_train <- qda_pred_train$class

evaluate_model_dual("QDA", y_test, qda_probs_test, qda_preds_test,
                    y_train, qda_probs_train, qda_preds_train)


#Decision Tree
library(rpart)
dt_model <- rpart(y ~ ., data = train_data, method = "class", control = rpart.control(cp = 0.01))

dt_probs_test <- predict(dt_model, newdata = test_data, type = "prob")[, "yes"]
dt_preds_test <- ifelse(dt_probs_test > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))

dt_probs_train <- predict(dt_model, newdata = train_data, type = "prob")[, "yes"]
dt_preds_train <- ifelse(dt_probs_train > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))

evaluate_model_dual("Decision Tree", y_test, dt_probs_test, dt_preds_test,
                    y_train, dt_probs_train, dt_preds_train)


#KNN
# Load library for KNN
library(class)
# 1. Select only numeric columns
x_train_numeric <- x_train[, sapply(x_train, is.numeric)]
x_test_numeric <- x_test[, sapply(x_test, is.numeric)]

# 2. Scale the numeric features
x_train_scaled <- scale(x_train_numeric)
x_test_scaled <- scale(x_test_numeric,
                       center = attr(x_train_scaled, "scaled:center"),
                       scale = attr(x_train_scaled, "scaled:scale"))

# 3. Fit KNN model (k = 5 as an example)
set.seed(123)
# Predict test
knn_pred_test <- knn(train = x_train_scaled, test = x_test_scaled, cl = y_train, k = 5)
knn_prob_test <- ifelse(knn_pred_test == "yes", 1, 0)

# Predict train (on same training data)
knn_pred_train <- knn(train = x_train_scaled, test = x_train_scaled, cl = y_train, k = 5)
knn_prob_train <- ifelse(knn_pred_train == "yes", 1, 0)

evaluate_model_dual("KNN (k=5)", y_test, knn_prob_test, knn_pred_test,
                    y_train, knn_prob_train, knn_pred_train)


### Model on Interaction terms ####
#############################################################
selected_cat_cols <- c("marital", "housing", "loan", "poutcome")
# Step 2: Separate numeric and categorical data
numeric_data <- train_data[, sapply(train_data, is.numeric)]
categorical_data <- train_data[, selected_cat_cols]

numeric_data_test <- test_data[, sapply(test_data, is.numeric)]
categorical_data_test <- test_data[, selected_cat_cols]

# Step 3: Numeric interactions (same as you did)
train_squared <- as.data.frame(sapply(numeric_data, function(x) x^2))
colnames(train_squared) <- paste0(colnames(numeric_data), "_squared")

test_squared <- as.data.frame(sapply(numeric_data_test, function(x) x^2))
colnames(test_squared) <- paste0(colnames(numeric_data_test), "_squared")

interact_names <- combn(names(numeric_data), 2, simplify = FALSE)
train_interactions_num <- as.data.frame(sapply(interact_names, function(pair) numeric_data[[pair[1]]] * numeric_data[[pair[2]]]))
colnames(train_interactions_num) <- sapply(interact_names, function(pair) paste(pair, collapse = "_x_"))

test_interactions_num <- as.data.frame(sapply(interact_names, function(pair) numeric_data_test[[pair[1]]] * numeric_data_test[[pair[2]]]))
colnames(test_interactions_num) <- sapply(interact_names, function(pair) paste(pair, collapse = "_x_"))

# Step 4: Categorical interactions
interact_names_cat <- combn(selected_cat_cols, 2, simplify = FALSE)

train_interactions_cat <- as.data.frame(sapply(interact_names_cat, function(pair) 
  interaction(categorical_data[[pair[1]]], categorical_data[[pair[2]]], drop = TRUE)))
colnames(train_interactions_cat) <- sapply(interact_names_cat, function(pair) paste(pair, collapse = "_x_"))

test_interactions_cat <- as.data.frame(sapply(interact_names_cat, function(pair) 
  interaction(categorical_data_test[[pair[1]]], categorical_data_test[[pair[2]]], drop = TRUE)))
colnames(test_interactions_cat) <- sapply(interact_names_cat, function(pair) paste(pair, collapse = "_x_"))

# Step 5: Combine all
# First save original target
train_y <- train_data$y
test_y <- test_data$y

# Now combine:
train_full <- cbind(train_data, train_squared, train_interactions_num, train_interactions_cat)
train_full$y <- train_y

test_full <- cbind(test_data, test_squared, test_interactions_num, test_interactions_cat)
test_full$y <- test_y

x_train <- train_full[, setdiff(names(train_full), "y")]
y_train <- train_full$y

x_test <- test_full[, setdiff(names(test_full), "y")]
y_test <- test_full$y

#Logistic Regression (with interaction terms)
logit_model_int <- glm(y ~ ., data = train_full, family = binomial)

# Predict Test
logit_probs_int <- predict(logit_model_int, newdata = test_full, type = "response")
logit_preds_int <- ifelse(logit_probs_int > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))

# Predict Train
logit_probs_int_train <- predict(logit_model_int, newdata = train_full, type = "response")
logit_preds_int_train <- ifelse(logit_probs_int_train > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))

evaluate_model_dual("Logistic Regression + Interactions", y_test, logit_probs_int, logit_preds_int,
                    train_full$y, logit_probs_int_train, logit_preds_int_train)


#Naive Bayes (with interaction terms)
nb_model_int <- naiveBayes(y ~ ., data = train_full)

nb_probs_int <- predict(nb_model_int, newdata = test_full, type = "raw")[, "yes"]
nb_preds_int <- predict(nb_model_int, newdata = test_full)

nb_probs_int_train <- predict(nb_model_int, newdata = train_full, type = "raw")[, "yes"]
nb_preds_int_train <- predict(nb_model_int, newdata = train_full)

evaluate_model_dual("Naive Bayes + Interactions", y_test, nb_probs_int, nb_preds_int,
                    train_full$y, nb_probs_int_train, nb_preds_int_train)

# LDA (with interaction terms)
lda_model_int <- lda(y ~ ., data = train_full)

lda_pred_int <- predict(lda_model_int, newdata = test_full)
lda_probs_int <- lda_pred_int$posterior[, "yes"]
lda_preds_int <- lda_pred_int$class

lda_pred_train_int <- predict(lda_model_int, newdata = train_full)
lda_probs_train_int <- lda_pred_train_int$posterior[, "yes"]
lda_preds_train_int <- lda_pred_train_int$class

evaluate_model_dual("LDA + Interactions", y_test, lda_probs_int, lda_preds_int,
                    train_full$y, lda_probs_train_int, lda_preds_train_int)



#QDA (with interaction terms)
# Select important numeric columns
important_vars <- c("age", "balance", "day", "duration", "campaign")

# Squared terms
train_squared_selected <- as.data.frame(sapply(train_data[, important_vars], function(x) x^2))
colnames(train_squared_selected) <- paste0(important_vars, "_squared")

test_squared_selected <- as.data.frame(sapply(test_data[, important_vars], function(x) x^2))
colnames(test_squared_selected) <- paste0(important_vars, "_squared")

# Select important interactions manually (for example: age x duration, balance x campaign)
train_interactions_selected <- data.frame(
  age_x_duration = train_data$age * train_data$duration,
  balance_x_campaign = train_data$balance * train_data$campaign
)

test_interactions_selected <- data.frame(
  age_x_duration = test_data$age * test_data$duration,
  balance_x_campaign = test_data$balance * test_data$campaign
)
# Combine back
train_qda_small <- cbind(train_data[, setdiff(names(train_data), "y")], 
                         train_squared_selected, 
                         train_interactions_selected)
train_qda_small$y <- train_data$y

test_qda_small <- cbind(test_data[, setdiff(names(test_data), "y")], 
                        test_squared_selected, 
                        test_interactions_selected)
test_qda_small$y <- test_data$y
# Fit QDA
qda_model_small <- qda(y ~ ., data = train_qda_small)

qda_pred_small <- predict(qda_model_small, newdata = test_qda_small)
qda_probs_small <- qda_pred_small$posterior[, "yes"]
qda_preds_small <- qda_pred_small$class

qda_pred_small_train <- predict(qda_model_small, newdata = train_qda_small)
qda_probs_small_train <- qda_pred_small_train$posterior[, "yes"]
qda_preds_small_train <- qda_pred_small_train$class

evaluate_model_dual("Simplified QDA (Selected Features)", test_qda_small$y, qda_probs_small, qda_preds_small,
                    train_qda_small$y, qda_probs_small_train, qda_preds_small_train)



#Decision tree (with interaction terms)
dt_model_int <- rpart(y ~ ., data = train_full, method = "class", control = rpart.control(cp = 0.01))

dt_probs_int <- predict(dt_model_int, newdata = test_full, type = "prob")[, "yes"]
dt_preds_int <- ifelse(dt_probs_int > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))

dt_probs_int_train <- predict(dt_model_int, newdata = train_full, type = "prob")[, "yes"]
dt_preds_int_train <- ifelse(dt_probs_int_train > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))

evaluate_model_dual("Decision Tree + Interactions", y_test, dt_probs_int, dt_preds_int,
                    train_full$y, dt_probs_int_train, dt_preds_int_train)


#KNN using interaction
# Select numeric columns for scaling
# Step 1: Keep only numeric columns (drop factors)
x_train_int_numeric <- x_train[, sapply(x_train, is.numeric)]
x_test_int_numeric <- x_test[, sapply(x_test, is.numeric)]

# Step 2: Scale properly
x_train_int_scaled <- as.data.frame(scale(x_train_int_numeric))
x_test_int_scaled <- as.data.frame(scale(x_test_int_numeric,
                                         center = attr(scale(x_train_int_numeric), "scaled:center"),
                                         scale = attr(scale(x_train_int_numeric), "scaled:scale")))

# Step 3: Train KNN
library(class)
set.seed(123)
knn_preds_int <- knn(train = x_train_int_scaled, test = x_test_int_scaled, cl = y_train, k = 5)
knn_prob_int <- ifelse(knn_preds_int == "yes", 1, 0)

knn_preds_int_train <- knn(train = x_train_int_scaled, test = x_train_int_scaled, cl = y_train, k = 5)
knn_prob_int_train <- ifelse(knn_preds_int_train == "yes", 1, 0)

evaluate_model_dual("KNN (k=5) + Interactions", y_test, knn_prob_int, knn_preds_int,
                    y_train, knn_prob_int_train, knn_preds_int_train)


#Model tuning
# Define control object for tuning
ctrl <- trainControl(
  method = "cv",        # Cross-validation
  number = 5,           # 5-fold CV
  classProbs = TRUE,    
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Original dataset
# Attach y_train and y_test to corresponding datasets
orig_train <- train_data  
orig_test <- test_data  
# Define datasets for interaction models
int_train <- train_full
int_test <- test_full

# Attach y_train and y_test
orig_train$y <- y_train
orig_test$y <- y_test

orig_train$y <- y_train
int_train$y <- y_train

orig_test$y <- y_test
int_test$y <- y_test

# Ensure correct factor format for binary classification
orig_train$y <- factor(orig_train$y, levels = c("no", "yes"))
orig_test$y <- factor(orig_test$y, levels = c("no", "yes"))
int_train$y <- factor(int_train$y, levels = c("no", "yes"))
int_test$y <- factor(int_test$y, levels = c("no", "yes"))

#Logistic Regression (Tuned via glmnet)
set.seed(101)
logit_orig <- train(y ~ ., data = orig_train,
                    method = "glmnet",
                    family = "binomial",
                    trControl = ctrl,
                    tuneGrid = expand.grid(alpha = 0, lambda = seq(0, 1, 0.05)),
                    metric = "ROC")

logit_pred <- predict(logit_orig, orig_test)
logit_prob <- predict(logit_orig, orig_test, type = "prob")[, "yes"]
evaluate_model("Logistic Regression (Tuned, Original)", orig_test$y, logit_prob, logit_pred)

#On Interaction Data
set.seed(101)
logit_int <- train(y ~ ., data = int_train,
                   method = "glmnet",
                   family = "binomial",
                   trControl = ctrl,
                   tuneGrid = expand.grid(alpha = 0, lambda = seq(0, 1, 0.05)),
                   metric = "ROC")

logit_pred_int <- predict(logit_int, int_test)
logit_prob_int <- predict(logit_int, int_test, type = "prob")[, "yes"]
evaluate_model("Logistic Regression (Tuned, Interaction)", int_test$y, logit_prob_int, logit_pred_int)

#Naive Bayes (Tuned)
#On Original Data
nb_grid <- expand.grid(laplace = 0:1, usekernel = c(TRUE, FALSE), adjust = 1:2)

set.seed(101)
nb_orig <- train(y ~ ., data = orig_train,
                 method = "naive_bayes",
                 trControl = ctrl,
                 tuneGrid = nb_grid,
                 metric = "ROC")

nb_pred <- predict(nb_orig, orig_test)
nb_prob <- predict(nb_orig, orig_test, type = "prob")[, "yes"]
evaluate_model("Naive Bayes (Tuned, Original)", orig_test$y, nb_prob, nb_pred)

#On Interaction Data
set.seed(101)
nb_int <- train(y ~ ., data = int_train,
                method = "naive_bayes",
                trControl = ctrl,
                tuneGrid = nb_grid,
                metric = "ROC")

nb_pred_int <- predict(nb_int, int_test)
nb_prob_int <- predict(nb_int, int_test, type = "prob")[, "yes"]
evaluate_model("Naive Bayes (Tuned, Interaction)", int_test$y, nb_prob_int, nb_pred_int)


#Decision Tree Tuning – Original Data
# Define grid for complexity parameter (cp)
dt_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))
set.seed(101)
dt_orig <- train(y ~ ., data = orig_train,
                 method = "rpart",
                 trControl = ctrl,
                 tuneGrid = dt_grid,
                 metric = "ROC")

# Predict and evaluate
dt_pred <- predict(dt_orig, orig_test)
dt_prob <- predict(dt_orig, orig_test, type = "prob")[, "yes"]

evaluate_model("Decision Tree (Tuned, Original)", orig_test$y, dt_prob, dt_pred)

#Decision Tree Tuning – Interaction Data
set.seed(101)
dt_int <- train(y ~ ., data = int_train,
                method = "rpart",
                trControl = ctrl,
                tuneGrid = dt_grid,
                metric = "ROC")

# Predict and evaluate
dt_pred_int <- predict(dt_int, int_test)
dt_prob_int <- predict(dt_int, int_test, type = "prob")[, "yes"]

evaluate_model("Decision Tree (Tuned, Interaction)", int_test$y, dt_prob_int, dt_pred_int)

# KNN Tuning – Original Data
# Only use numeric predictors for KNN
x_train_orig_knn <- orig_train[, sapply(orig_train, is.numeric)]
x_test_orig_knn <- orig_test[, sapply(orig_test, is.numeric)]
y_train_orig_knn <- orig_train$y
y_test_orig_knn <- orig_test$y

# Create a combined dataset to make it compatible with caret
train_orig_knn <- cbind(x_train_orig_knn, y = y_train_orig_knn)
test_orig_knn <- cbind(x_test_orig_knn, y = y_test_orig_knn)

# Tune k values
set.seed(101)
knn_orig <- train(y ~ ., data = train_orig_knn,
                  method = "knn",
                  trControl = ctrl,
                  tuneGrid = expand.grid(k = seq(3, 15, 2)),
                  metric = "ROC")

# Predict and evaluate
knn_pred_orig <- predict(knn_orig, test_orig_knn)
knn_prob_orig <- predict(knn_orig, test_orig_knn, type = "prob")[, "yes"]

evaluate_model("KNN (Tuned, Original)", y_test_orig_knn, knn_prob_orig, knn_pred_orig)

# --------------------------------------------------------------------------------------------------

# KNN Tuning – Interaction Data
# Only use numeric predictors for KNN
x_train_int_knn <- int_train[, sapply(int_train, is.numeric)]
x_test_int_knn <- int_test[, sapply(int_test, is.numeric)]
y_train_int_knn <- int_train$y
y_test_int_knn <- int_test$y

# Combine for caret
train_int_knn <- cbind(x_train_int_knn, y = y_train_int_knn)
test_int_knn <- cbind(x_test_int_knn, y = y_test_int_knn)

# Tune
set.seed(101)

# Predict and evaluate
knn_pred_int <- predict(knn_orig, test_int_knn)
knn_prob_int <- predict(knn_orig, test_int_knn, type = "prob")[, "yes"]

evaluate_model("KNN (Tuned, Interaction)", y_test_int_knn, knn_prob_int, knn_pred_int)

####################PCA On full data###################################

# Step 1: One-hot encode the train and test datasets
train_encoded <- as.data.frame(model.matrix(~ . -1, data = train_data))
test_encoded <- as.data.frame(model.matrix(~ . -1, data = test_data))

# Step 2: Standardize the data
train_scaled <- scale(train_encoded)
test_scaled <- scale(test_encoded,
                     center = attr(train_scaled, "scaled:center"),
                     scale = attr(train_scaled, "scaled:scale"))

# Step 3: PCA
pca <- prcomp(train_scaled, center = FALSE, scale. = FALSE)
cum_var <- cumsum((pca$sdev)^2 / sum((pca$sdev)^2))
num_pc_95 <- which(cum_var >= 0.95)[1]
cat("Number of PCs to retain 95% variance:", num_pc_95, "\n")

x11()
plot(cum_var, type = "b", xlab = "Number of Principal Components",
     ylab = "Cumulative Proportion of Variance Explained",
     main = "PCA - Variance Explained")
abline(h = 0.95, col = "red", lty = 2)
abline(v = num_pc_95, col = "blue", lty = 2)
# Step 4: Create PCA-transformed data
train_pca <- as.data.frame(pca$x[, 1:num_pc_95])
test_pca <- as.data.frame(predict(pca, newdata = test_scaled)[, 1:num_pc_95])

# Add target variable back
train_pca$y <- train_data$y
test_pca$y <- test_data$y

# Split features and target
x_train <- train_pca[, setdiff(names(train_pca), "y")]
y_train <- train_pca$y
x_test <- test_pca[, setdiff(names(test_pca), "y")]
y_test <- test_pca$y

# ------------------------------------------
# Logistic Regression (PCA)
logit_pca <- glm(y ~ ., data = train_pca, family = binomial)
logit_prob_pca <- predict(logit_pca, newdata = test_pca, type = "response")
logit_pred_pca <- ifelse(logit_prob_pca > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))
evaluate_model("Logistic Regression (PCA)", y_test, logit_prob_pca, logit_pred_pca)

# ------------------------------------------
# Naive Bayes (PCA)
library(e1071)
nb_pca <- naiveBayes(y ~ ., data = train_pca)
nb_prob_pca <- predict(nb_pca, newdata = test_pca, type = "raw")[, "yes"]
nb_pred_pca <- predict(nb_pca, newdata = test_pca)
evaluate_model("Naive Bayes (PCA)", y_test, nb_prob_pca, nb_pred_pca)

# ------------------------------------------
# LDA (PCA)
library(MASS)
lda_pca <- lda(y ~ ., data = train_pca)
lda_pred_pca <- predict(lda_pca, newdata = test_pca)
evaluate_model("LDA (PCA)", y_test, lda_pred_pca$posterior[, "yes"], lda_pred_pca$class)

# ------------------------------------------
# QDA (PCA)
qda_pca <- qda(y ~ ., data = train_pca)
qda_pred_pca <- predict(qda_pca, newdata = test_pca)
evaluate_model("QDA (PCA)", y_test, qda_pred_pca$posterior[, "yes"], qda_pred_pca$class)

# ------------------------------------------
# Decision Tree (PCA)
library(rpart)
dt_pca <- rpart(y ~ ., data = train_pca, method = "class", control = rpart.control(cp = 0.01))
dt_prob_pca <- predict(dt_pca, newdata = test_pca, type = "prob")[, "yes"]
dt_pred_pca <- ifelse(dt_prob_pca > 0.5, "yes", "no") |> factor(levels = c("no", "yes"))
evaluate_model("Decision Tree (PCA)", y_test, dt_prob_pca, dt_pred_pca)

# ------------------------------------------
# KNN (PCA)
library(class)
k <- 5
knn_pred_pca <- knn(train = x_train, test = x_test, cl = y_train, k = k)

# For KNN, assume probability as 1 if prediction is 'yes', 0 otherwise
knn_prob_pca <- ifelse(knn_pred_pca == "yes", 0, 1)
evaluate_model(paste("KNN (k=", k, ") (PCA)", sep = ""), y_test, knn_prob_pca, knn_pred_pca)
