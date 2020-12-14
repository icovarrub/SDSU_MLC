library(caret)
library(lattice)
library(ggplot2)
library(MASS)
library(e1071)
library(psych)
library(corrplot)
library(pROC)
library(ISLR)

setwd("C:/Users/covarrubias/Documents/749_Project")
mlc <- readRDS("Imputed-MLC-data.RDS")
str(mlc)
names(mlc)

##PREPROCESSING------------------
#convert factors to dummy coded predictors ahead of time

#using caret create a model of the dummy variables, with Full rank so N factor levels-1 new predictors
mlc.dmodel <- dummyVars( ~ ., data=mlc, fullRank=T)

#apply the model to create the new variables
#and mlc new data frame with dummy codes instead of factors
mlc.d <- as.data.frame(predict(mlc.dmodel, mlc))
str(mlc.d)
head(mlc.d)

#Remove highly correlated variables
mlc.d.cor <- cor(mlc.d)
mlc.d.cor

highcor <- findCorrelation(mlc.d.cor, cutoff=.8)
highcor
names(mlc.d.cor[,highcor])

mlc.d.final <- mlc.d[,-highcor]
mlc.d.final
names(mlc.d.final)

#Convert DFW outcome to a factor with values "yes" and "no"
mlc.d.final$DFW <- as.factor(mlc.d.final$DFW)
mlc.d.final$DFW <- ifelse(mlc.d.final$DFW==1,'yes','no')
mlc.d.final$DFW <- as.factor(mlc.d.final$DFW)

#is there unbalanced data?
mlc.d.table<- table(mlc.d$DFW)
prop.table(mlc.d.table) # .29% of cases are positive class

#Relevel the DFW factor so "yes" becomes the positive class by default (Sensitivity)
mlc.d.final$DFW <- relevel(mlc.d.final$DFW, ref="yes")

#NULL Predictors with singularities
mlc.d.final$online <- NULL
mlc.d.final$FirstGen_BachDegree <- NULL
mlc.d.final$Admit_Basis_Code.L <- NULL

#split into train and test sets - keep 70% of data for training
set.seed(195)
train.index <- sample(nrow(mlc.d.final), nrow(mlc.d.final) * .7)

#create test and training data frames
mlc.train <- mlc.d.final[train.index,] #model with this
mlc.test <- mlc.d.final[-train.index,] #we don't touch this while training

#double check yes and no counts with training data
mlc.train.table <- table(mlc.train$DFW)
prop.table(mlc.train.table) #still .29%!

###RUN MODELS

#saving predictions from each resample fold
ctrl <- trainControl(method = "cv", number=10, summaryFunction=twoClassSummary,
                     classProbs=T, savePredictions=T)

##run glm using logistic regression model
set.seed(195)
mlc.log <-  train(DFW ~ ., data=mlc.train, method="glm", family="binomial", metric="ROC", trControl=ctrl)
summary(mlc.log) #model summary
varImp(mlc.log) #rank important variables
mlc.log

#calculate resampled accuracy/confusion matrix using extracted predictions from resampling
confusionMatrix(mlc.log$pred$pred, mlc.log$pred$obs) #take averages

#ROC curve
mlc.log.roc <- roc(response= mlc.log$pred$obs, predictor=mlc.log$pred$yes)
plot(mlc.log.roc, legacy.axes=T)

##Run LDA
set.seed(195)
mlc.lda <- train(DFW ~ ., data=mlc.train, method="lda", metric="ROC", trControl=ctrl)
mlc.lda
varImp(mlc.lda)
confusionMatrix(mlc.lda$pred$pred, mlc.lda$pred$obs) #take averages

#ROC curve
mlc.lda.roc<- roc(response= mlc.lda$pred$obs, predictor=mlc.lda$pred$yes)
plot(mlc.lda.roc, legacy.axes=T)

#k nearest neighbors classification
set.seed(195) 
mlc.knn <-  train(DFW ~ ., data=mlc.train, method="knn", metric="ROC", trControl=ctrl, tuneLength=10) #let caret decide 10 best parameters to search
mlc.knn
varImp(mlc.knn)
confusionMatrix(mlc.knn$pred$pred, mlc.knn$pred$obs) #take averages

#ROC curve
mlc.knn.roc<- roc(response= mlc.knn$pred$obs, predictor=mlc.knn$pred$yes)
plot(mlc.knn.roc, legacy.axes=T)

##Logistic Lasso/Ridge Regression, alpha = ridge vs lasso ratio, lambda=shrinkage amount
library(glmnet)
set.seed(195)
mlc.glmnet <- train(DFW ~ ., data=mlc.train, method="glmnet", metric="ROC", trControl=ctrl)
mlc.glmnet
varImp(mlc.glmnet)
confusionMatrix(mlc.glmnet$pred$pred, mlc.glmnet$pred$obs) #take averages

#Isolate best tune
mlc.glmnet$bestTune

mlc.glmnet.pred<- predict(mlc.glmnet, s=mlc.glmnet$bestTune$lambda, mlc.train, type="prob")

mlc.glmnet.pred[2]

mlc.glmnet.bt <- subset(mlc.glmnet$results, 
                        alpha == mlc.glmnet$bestTune$alpha & lambda== mlc.glmnet$bestTune$lambda)

##Decision tree
set.seed(195)
mlc.rpart <- train(DFW ~ ., data=mlc.train, method="rpart", tuneLength=4, trControl=ctrl)
mlc.rpart
varImp(mlc.rpart)
confusionMatrix(mlc.rpart$pred$pred, mlc.rpart$pred$obs) #take averages

##Bagging
set.seed(195)
mlc.bag <- train(DFW ~ ., data=mlc.train, method="treebag",tuneLength=4, trControl=ctrl)
mlc.bag
varImp(mlc.bag)
confusionMatrix(mlc.bag$pred$pred, mlc.bag$pred$obs) #take averages

##Boosting
set.seed(195)
mlc.boost <- train(DFW ~ ., data=mlc.train, method="gbm", tuneLength=4, trControl=ctrl)
mlc.boost
varImp(mlc.boost)
confusionMatrix(mlc.boost$pred$pred, mlc.boost$pred$obs) #take averages

#Isolate best tune
mlc.boost$bestTune

###Lets look at all training model performances
getTrainPerf(mlc.log)
getTrainPerf(mlc.lda)
getTrainPerf(mlc.knn)
getTrainPerf(mlc.glmnet)
getTrainPerf(mlc.rpart)
getTrainPerf(mlc.bag)
getTrainPerf(mlc.boost)

#lets compare all resampling approaches
mlc.models <- list("LOG"=mlc.log, "LDA"=mlc.lda, "KNN"=mlc.knn, "GLMNET"=mlc.glmnet, "DT"=mlc.rpart, "BAG"=mlc.bag, "BOOST"=mlc.boost)
mlc.resamples = resamples(mlc.models)

#plot performance comparisons
bwplot(mlc.resamples, metric="ROC") 
bwplot(mlc.resamples, metric="Sens") #predicting default dependant on threshold
bwplot(mlc.resamples, metric="Spec")

#plot performance comparisons - scaled x-axis
bwplot(mlc.resamples, scales=list(relation="free"), xlim=list(c(0,1),c(0,1),c(0,1)), metric="ROC") 
bwplot(mlc.resamples, scales=list(relation="free"), xlim=list(c(0,1),c(0,1),c(0,1)), metric="Sens") #predicting default dependant on threshold
bwplot(mlc.resamples, scales=list(relation="free"), xlim=list(c(0,1),c(0,1),c(0,1)), metric="Spec")

##LDA have the lowest Error Rates - the best choice
##Move forward and identify optimal cut-off with LDA model (current resampled confusion matrix); low sensitivity
confusionMatrix(mlc.lda$pred$pred, mlc.lda$pred$obs)

#extract threshold from roc curve get threshold at coordinates top left most corner
mlc.lda.Thresh <- coords(mlc.lda.roc, x="best", best.method="closest.topleft")
mlc.lda.Thresh #sensitivity increases to 64% by reducing threshold to .2910 from .5

#lets make new predictions with this cut-off and recalculate confusion matrix
mlc.lda.newpreds <- factor(ifelse(mlc.lda$pred$yes > mlc.lda.Thresh[1], "yes", "no"))

#recalculate confusion matrix with new cut off predictions
confusionMatrix(mlc.lda.newpreds, mlc.lda$pred$obs)

### TEST DATA PERFORMANCE
#lets see how this cut off works on the test data
#predict probabilities on test set with LDA trained model
test.pred.prob <- predict(mlc.lda, mlc.test, type="prob")

test.pred.class <- predict(mlc.lda, mlc.test) #predict classes with default .5 cutoff

#calculate performance with confusion matrix
confusionMatrix(test.pred.class, mlc.test$DFW)

#calculate test confusion matrix using thresholds from resampled data
test.pred.class.newthresh <- factor(ifelse(test.pred.prob[[1]] > mlc.lda.Thresh[1], "yes", "no"))

#recalculate confusion matrix with new cut off predictions
confusionMatrix(test.pred.class.newthresh, mlc.test$DFW)

#Thresholds MUST be adjusted when dealing with unbalanced data
#Falsely identifying failing students may be expensive and a waste of resources
