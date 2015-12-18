#start

```{r}
cat("Init");
setwd("Q:\\dev\\coursera\\ml");

#libraries
library(caret)
library(kernlab)
library(ggplot2)
library(lattice)
library(corrplot)
library(tree)
library(rattle)
library(rpart)
library(rpart.plot)
require(randomForest)

#tuning and settings
OUTPUT_EXPLORATORY_PLOTS <- FALSE
TEST_ROWS <- 1000 #if > -1, test code with a subset of observations
TRAINING_PCT <- 0.8
CORREL_THRESHHOLD <- 0.95

#seed
set.seed(33833)

#load data
pml_training = read.csv("pml-training.csv", header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))
pml_testing = read.csv("pml-testing.csv", header = TRUE, stringsAsFactors = FALSE, na.strings=c("", "NA", "NULL"))
if (TEST_ROWS > -1) {
  pml_training <- pml_training[sample(nrow(pml_training), TEST_ROWS), ]
}
  
dim(pml_training)
#summary(pml_training)
#str(pml_training)

#tidying of training set
pml_training_tidy <- pml_training
#change classe into factor
pml_training_tidy$classe = as.factor(pml_training_tidy$classe)
#Some sparse fields are incorrectly interpreted as characters, turn them explicitly into numerics
transform_fields <- grep("^(kurtosis|max|min|skewness)", colnames(pml_training_tidy))
pml_training_tidy[transform_fields] <- lapply(pml_training_tidy[transform_fields], as.numeric)
#remove mostly NA fields
pml_training_tidy <- pml_training[ , colSums(is.na(pml_training)) == 0]
#remove non-data fields
nondata_fields = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
pml_training_ind <- pml_training_tidy[, -which(names(pml_training_tidy) %in% nondata_fields)]

#exploratory analysis - output boxplots for all variables
if (OUTPUT_EXPLORATORY_PLOTS) {
  for (ind in 1:(ncol(pml_training_ind) - 1)) {
    png(paste0("img/box_", colnames(pml_training_ind)[ind], ".png"))
    p <- ggplot(
      pml_training_ind, 
      aes_string(x = "classe", y = colnames(pml_training_ind)[ind], fill = "classe")
    ) + geom_boxplot() #+ ylim(-5, 5)
    print(p)
    dev.off()    
  }
}

#yaw_belt has good explanatory power. Intuitively, We expect this to show up in analysis
p <- ggplot(
  pml_training_ind, 
  aes(x = classe, y = total_accel_belt, fill = classe)
) + geom_boxplot() #+ ylim(-5, 5)
print(p)

#the following variables are also good indicators. 
primary_indep_vars <- c(
  "accel_arm_x", 
  "accel_belt_y", 
  "accel_belt_z",
  "amplitude_pitch_belt",
  "amplitude_pitch_forearm",
  "avg_roll_belt",
  "avg_roll_dumbbell",
  "avg_yaw_belt",
  "magnet_arm_x",
  "max_picth_arm",
  "max_picth_belt",
  "max_picth_dumbbell",
  "max_roll_belt",
  "max_roll_dumbbell",
  "min_pitch_belt",
  "min_pitch_forearm",
  "min_roll_belt",
  "min_roll_forearm",
  "roll_belt",
  "roll_forearm",
  "stddev_pitch_belt",
  "stddev_roll_belt",
  "stddev_roll_forearm",  
  "stddev_yaw_forearm",
  "total_accel_belt",
  "var_pitch_dumbbell",
  "var_roll_belt",
  "var_roll_forearm",
  "var_total_accel_belt",
  "var_yaw_forearm"
)
primary_indep_vars

#study correlation of covariates
corrMatrix <- cor(na.omit(pml_training_ind[sapply(pml_training_ind, is.numeric)]))
corrplot(corrMatrix, order="AOE", method="pie", tl.cex=0.6)  

#and remove highly correlated
remove_highcorrelation <- findCorrelation(corrMatrix, CORREL_THRESHHOLD, verbose=TRUE)
pml_training_independent = pml_training_ind[,-remove_highcorrelation]

#align training/testing sets
pml_testing_independent <- pml_testing[, which(names(pml_testing) %in% names(pml_training_independent))]

#how many fields do we have left?
dim(pml_training_independent)
dim(pml_testing_independent)

#split original training set into test / training for CV
inTrain <- createDataPartition(y=pml_training_independent$classe, p=TRAINING_PCT, list=FALSE)
training <- pml_training_independent[inTrain,]
testing <- pml_training_independent[-inTrain,]

#use several ML methods 
ml_methods <- c("rf", "gbm", "lda")
ensemble_pred <- data.frame(NA, NA, NA, classe=testing$classe)
pml_testing_pred <- data.frame(rep(NA, nrow(pml_testing_independent)), NA, NA, NA)
colnames(ensemble_pred) = c(ml_methods, "classe")
colnames(pml_testing_pred) = colnames(ensemble_pred)

#for each method
for (ml_method in ml_methods) {
  cat ("Running ML Algorithm", ml_method, ":\n")
  
  #train 
  fitML <- train (
    classe ~ ., 
    data=training, 
    method=ml_method,
    metric = "Accuracy",
    #preProcess=c("pca"), 
    trControl = trainControl(method = "cv"),
    verbose = FALSE
  )
  
  #output variable importance - for some reason this doesn't work in lda
  if (ml_method != "lda") {
    p <- plot(varImp(fitML, scale = FALSE), top=10)
    print(p)
  }
  
  #predict
  pred <- predict(fitML, testing)
  ensemble_pred[ml_method] <- pred

  #accuracy
  cm <- confusionMatrix(pred, testing$classe)
  cat ("Accuracy:", cm$overall[1], "\n")

  #apply model to final testing set
  pred <- predict(fitML, pml_testing_independent)
  pml_testing_pred[ml_method] <- pred
}

#stack (ensemble) models
final_fit <- train(classe ~., data=ensemble_pred, method="rf")
predFit <- predict(final_fit, ensemble_pred)
cm <- confusionMatrix(predFit, ensemble_pred$classe)
cat ("Ensemble Accuracy:", cm$overall[1], "\n")

# predict final training
pml_testing_pred$classe <- predict(final_fit, pml_testing_pred)
pml_testing_pred
```
