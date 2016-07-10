# PROTOTYPING WORKSHEET (WORK-IN-PROGRESS) FOR MACHINE LEARNING PROJECT

# PACKAGE/LIBRARY LOADER
usePackage<-function(p){
  # load a package if installed, else load after installation.
  # Args: p: package name in quotes
  if (!is.element(p, installed.packages()[,1])){
    print(paste('Package:',p,'Not found, Installing Now...'))
    install.packages(p, dep = TRUE)}
  print(paste('Loading Package :',p))
  require(p, character.only = TRUE)  
}

usePackage("caret")
usePackage("ggplot2")
usePackage("rattle")
usePackage("randomForest")
usePackage("rpart")
usePackage("e1071")

# REVIEW SESSION
sessionInfo()

# SET WORKING DIRECTORY
setwd("/Users/smallwes/develop/academic/coursera/datascience/c8-ml/project1/")

# PROJECT DATA FILES (ASSUMING DOWNLOADED FILES)
trainingFilename <- "pml-training.csv"
testingFilename <- "pml-testing.csv"
#trainingFile <- url("https://d396qusza40orc.cloumodelDFront.net/predmachlearn/pml-training.csv")
#testingFile <- url("https://d396qusza40orc.cloumodelDFront.net/predmachlearn/pml-testing.csv")

# READ DATASETS
trainingDataset <- read.csv(trainingFilename) # NOTE: This dataset will be partitioned into model training/testing subsets.
quizTestingDataset <- read.csv(testingFilename)

# INITIAL INITIAL REVIEW TRAINING DATA SET
dim(trainingDataset)
dim(quizTestingDataset)
#View(training)
str(trainingDataset)
#summary(training)
#head(training)
#str(testing)

# ----------------
# SUMMATION OF BASE DATASET FINDINGS:

# "classe" column is located at the end; column 153
# Model creation sees us relating the outcome to "all other columns" using formula "classe ~ ."
# However a significant number of fields have invalid data in numeric columns - standardize "NA" data upon read.
# A tidy-data phase shall occur identically on both training and the quiz testing dataset before before we engage
# in model building (cross validation, prediction, and comparison of models)
#
# Initial steps:
# Create dataframe with NA data specify na data: "NA", blanks, and unknown numeric values "#DIV/0!"
# Eliminate columns with no content including columns with NAs (i.e colmns max_roll_arm : num  NA NA NA NA NA NA NA NA NA NA ...)
# Remove columns with non helpful model building info, i.e. X, username, new window, time columns as we're not forecasting
# Given the notes from the ML course, we could look at preprocessing the data with imputting, eliminating near-zero values, etc.
# -----------------

# DEFINE EMPTY DATA SIGNATURES
naStringList = c("", "NA","#DIV/0!")

# RELOAD WITH NA COLUMNS SETTING (Assume WK/Directory Set)
modelDF <- read.csv(trainingFilename, na.strings=naStringList, header=TRUE)
dim(modelDF)

# SUBSET DF CLASS REQUIRED
classeColumn <- modelDF$classe

# REORDER COLUMNS ( MOVE CLASSE COLUMN TO START)
modelDF <- modelDF[,c(ncol(modelDF),1:(ncol(modelDF)-1))]
str(modelDF)
#head(modelDF, 1)
#dim(modelDF)

# EXPLICIT REMOVAL OF COLUMNS (NON HELPFUL; NOT NECESSARY FOR MODEL CREATION)
colnames(modelDF)[c(2:8)]
modelDF <- modelDF[,-c(2:8)]
#str(modelDF)
dim(modelDF)

# REMOVE COLUMN WITH NAs
# Review: http://stackoverflow.com/questions/2643939/remove-columns-from-dataframe-where-all-values-are-na
modelDF <- modelDF[, unlist( lapply( modelDF, function(x) { !all(is.na(x) ) } ) ) ]
dim(modelDF)

getAnyNAColumnIndices <- function(modelDF) {
 output <- list(rep(FALSE, ncol(modelDF)))
 for (i in 1:ncol(modelDF)) {
   output[i] <- (length( which(is.na(modelDF[,i]))) !=0)
 }
 unlist(output)
}
#getAnyNAColumnIndices(modelDF)
modelDF <- modelDF[,!getAnyNAColumnIndices(modelDF)]
#str(modelDF)
dim(modelDF)

# PREPROCESSING ( ML WK #2 LECTURE: BASIC PREPROCESSING )
# A) STANDARDIZATION OF PREDICTORS VARIABLES ( MINUS MEAN DIVIDED BY STANDARD DEVIATION)
# B) PRE-PROCESS FUNCTIONS
#preObj <- preProcess(modelDF[, -1], method=c("knnImpute", "center","scale" ))
#modelDF <- predict(preObj, modelDF[,-1])
#modelDF$classe <- classeColumn

# DECIDED TO PERFORM STANDARDIZATION WHEN TRAINING MODEL;
# i.e. train(modelDF, method="XXXX", preProcess=c("knnImpute", "center","scale" ))

# REMOVING ZERO COVARIATES ( ML WK #2 LECTURE: COVARIATE CREATION )
nzvDF <- nearZeroVar(modelDF, saveMetric=TRUE, names=TRUE)
isAnyNZV <-any(nzvDF$nzv)
if (isAnyNZV) {
  modelDF <- modelDF[, -nzvDF]
}
dim(modelDF)
colnames(modelDF)

# REMOVE HIGHLY CORRELATED PREDICTORS ( ML WK #2 LECTURE: PREPROCESSION WITH PCA )
# Method in comment: http://stackoverflow.com/questions/18275639/remove-highly-correlated-variables
#corDF <- abs(cor(modelDF[,-1]))
#diag(corDF) <- 0
#which(M > 0.8, arr.ind = T)
corDF <- cor(modelDF[,-1])
hc <- findCorrelation(corDF, cutoff=0.8) #high correlated predictors matrix
hc <- sort(hc)
modelDF <- modelDF[,-c(hc + 1) ]
#modelDF <- rbind(classeColumn, modelDF[,-c(hc)]) #
#modelDF <- modelDF[,c(ncol(modelDF),1:(ncol(modelDF)-1))] # classe as first col again.
dim(modelDF)
colnames(modelDF)

# REVIEW PRINCIPAL COMPONENT ANALYSIS ( ML WK #2 LECTURE: PREPROCESSING WITH PCA )
#prComp <- prcomp(log10(modelDF[,-1]+1))

# REVIEW PREDICTING WITH LINEAR REGRESSION & MULTIPLE COVARIATES
# Not applicable for this project"
#featurePlot(x=training[,-1], y=modelDF$classe, plot="pairs")
#qplot(x,y, colour=classe, data=modelDF)
#plot(finalMod,1, pch=19,cex=0.5,col="#00000010") # diagnostics
#qplot(finalMod$fitted,finalMod$residuals,colour=classe,data=modelDF)
#plot(finalMod$residals, pch=19)  # plot by index


# MODEL CHOICE: CROSS VALIDATION SUBSET TRAINING & TESTING 
set.seed(1234)
inTrain = createDataPartition(modelDF$classe, p = 0.6, list=FALSE)
training = modelDF[ inTrain,]
testing = modelDF[-inTrain,]
dim(training)
dim(testing)

# MODEL TRAINING BY TAG
#?train
# Review Model: http://topepo.github.io/caret/bytag.html

# COMPARISON OF MODELS:

# TECHNIQUE: http://machinelearningmastery.com/compare-models-and-select-the-best-using-the-caret-r-package/
# PREPARE TRAINING SCHEMES:
paramFormula <- classe ~ .
paramControl <- trainControl(
                    method = "repeatedCV", number = 10, repeats = 5, 
                    returnResamp = "all", classProbs = TRUE)
paramPreProcess <- c("knnImpute", "center","scale")

# TRAIN: RPART (using caret train)
set.seed(1234)
modelRpartA <- train(paramFormula, method="rpart", data=training, preProcess=paramPreProcess)

# TRAIN: RPART (using rpart)
set.seed(1234)
modelRpartB <- rpart(paramFormula, data=training, method="class")  # Outcome y is factor A -> E

# TRAIN: Glmnet
set.seed(1234)
modelGlmnet <- train(paramFormula, data=training, method="glmnet", metric="ROC", tuneGrid=expand.grid(.alpha = c(0,1), .lambda = seq(0, 0.05, by = 0.01)), trControl=paramControl)

# TRAIN: SVM-RADICAL
set.seed(1234)
modelSvm <- train(paramFormula, data=training, method="svmRadial")

# TRAIN: RF / RANDOM FOREST
#modelRF <- train(paramFormula, method="rf", data=training, 
#                  preProcess=c("knnImpute", "center","scale"), 
#                  trControl=trainControl(method="cv",number=5),
#                  prox=TRUE,allowParallel=TRUE)
# ABORT training model with metho "rf" = reason taking too long to complete.
# http://stats.stackexchange.com/questions/37370/random-forest-computing-time-in-r
# Switch to specific random forest library "randomForest"
modelRandomForests <- randomForest(paramFormula, data=training,  proximity=TRUE, keep.forest=TRUE, importance=TRUE)

# COLLECT RESAMPLES
#results <- resamples(list(RPART_A=modelRpartA, RPART_B=modelRpartA, LVQ=modelLvq, GBM=modelGbm, SVM=modelSvm, RF=modelRandomForests))

# SUMMARIZE DISTRIBUTIONS
#summary(results)

# PLOTS OF SUMMARIES MODEL COMPARISION AGAINS TRAINING DATA
#bwplot(results)
#dotplot(results)

# TREE FIGURE: USING RPART-A
fancyRpartPlot(modelRpartA$finalModel)

# TREE FIGURE: USING RPART-B
fancyRpartPlot(modelRpartB)

# PLOT SVM
plot(modelSvm,training)

# PLOTS FOR RANDOM FOREST
plot(modelRandomForests, log="y")
varImpPlot(modelRandomForests)
#MDSplot(modelRandomForests, testing$classe) # Crashed; takes to long
#rfTree <- getTree(modelRandomForests,1,labelVar=TRUE)
#dendrogramRandomForests <- to.dendrogram(rfTree)
#str(dendrogramRandomForests)
#plot(dendrogramRandomForests,center=TRUE,leaflab='none',edgePar=list(t.cex=1,p.col=NA,p.lty=0))

# PREDICTIONS AGAINST RPART-B
predRpartB <- predict(modelRpartB, testing, type="class")
accuracyRpartA <- confusionMatrix(predRpartB,testing$classe)$overall['Accuracy']

# PREDICTIONS AGAINST RANDOM-FORESTS
predRandomForests <- predict(modelRandomForests, testing)
confusionMatrix(predRandomForests,testing$classe)
confusionMatrix(predRandomForests,testing$classe)$overall['Accuracy']


# ----------------------
# PREDICT PROJECT QUIZ ANSWERS:

# LOAD QUIZ TESTING DATASET
quizTestingDF <- read.csv(testingFilename, na.strings=naStringList, header=TRUE)

# SUBSET COLUMNS OF QUIZ DATASET FROM THE COLUMNS REDUCED IN MODEL-DF
quiz <- quizTestingDF[, which(names(quizTestingDF) %in% colnames(modelDF))]
dim(quiz)

# NO CLASSE COLUMN -> WE'LL PREDICT!

# PERFORM PREDICTIONS ON QUIZ TESTING DATASET
predict(modelRandomForests, quiz)

#quizPredictions
#1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
#B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
#Levels: A B C D E