##############################################################
### Projet Machine Learning - Elman NN -  la serie sunspot ###
################ author: BEN ZEKRI Nouriddin #################
##############################################################

## Les packages necessaires ###

require(caret)
require(fpp)
require (quantmod)
require(RSNNS)

######## data partitioning ########

seasonplot(sunspot, col='blue')
x1 <- ts(sunspotarea)
plot(x1, main="sunspot dataset", ylab = "sunspot")
View(x1)
boxplot(x1, col = 'blue')
hist(x1,prob=TRUE, col="grey",main= 'Distribution et Densité de la série X1')# prob=TRUE for probabilities not counts
lines(density(x1), col="blue", lwd=2)
summary(x1)

samp <- data.frame(x1)   # convert x1 time-series object to data.frame
n <- nrow(samp) # le nombre les observations
train_i <- 1:round(0.7 * n) # 70% training set
valid_i <- round(0.7 * n + 1):round(0.85 * n) # 15 % validation set
test_i <- round(0.85 * n + 1):n # 15 % test set
train <- samp[train_i, ]
valid <- samp[valid_i, ]
test <- samp[test_i, ]
# Convert the partitions to rime-serie
trainingSet <- ts(train)
validSet <- ts(valid)
testSet <- ts(test)
head(x1)
# Normaliser les données avec la methode MinMAx #
min_data <- min(x1)
max_data <- max(x1)
MinMax_data <- function (x) {
  (x - min_data) / (max_data - min_data)
}
trainingSet.nor <- MinMax_data (trainingSet)
validSet.nor <- MinMax_data (validSet)
testSet.nor <- MinMax_data (testSet)
plot(sunspotarea, main = "Sunspot data")
##########################
# -----ELMAN MODEL------ #
##########################

## preparer les entrées de notre modèle
require (quantmod)
split_lag_data <- function (data) {
  y <- as.zoo(data)
  x0 <- Lag(y ,  k = 1)
  x2 <- Lag(y ,  k = 2)
  x3 <- Lag(y ,  k = 3)
  x <- cbind (x0, x2, x3)
  x <- cbind (y, x)
  x <- x [-(1:3),]
  return(x)
}
x <- split_lag_data(trainingSet.nor)
View(trainingSet.nor)
head(x)
plot(x, main='training data')
n = nrow (ts(x))
n
##### Train the Elman Model with the Data #####
set.seed (450)
inputs <- x [, 2:4] # colonnes Lag.1, Lag.2 et Lag.3 
outputs <- x [, 1] # La sortie y
require (RSNNS)
test <- split_lag_data(testSet.nor)
testInput <- test[, 2:4]
testOutput <- test[,1]

fit <- elman (inputs,
              outputs ,
              learnFunc = "JE_BP",
              initFunc = "JE_Weights",
              linOut = TRUE,
              inputsTest = testInput ,
              targetsTest = testOutput,
              learnFuncParams = c(0.1) ,
              size = c (2,2) ,  
              maxit = 2000)

##### Evaluating Model Performance #####

plotIterativeError(fit, col='red') #Erreur iterative
plotRegressionError(outputs, fit$fitted.values) # erreur de regression

# erreur de training #
round(cor(outputs, fit$fitted.values) ^ 2, 4)
round(cor(outputs, fit$fitted.values), 4)
MAE(outputs, fit$fitted.values)
RMSE(outputs, fit$fitted.values)
mse(outputs,fit$fitted.values )
#                    #
# tester le modele sur les données de validation

dataV <- split_lag_data(validSet.nor)
indataV <- dataV[,2:4]
outdataV <- dataV[,1]
pred <- predict(fit, indataV)
plot(outdataV, type = 'line')
lines(pred, col='red')

round(cor(outdataV, pred) ^ 2, 4)
round(cor(outdataV, pred), 4)
MAE(outdataV, pred)
RMSE(outdataV, pred)
mse(outdataV,pred )

############ Predict ############
### tester le model sur toute la série
data1 <- split_lag_data(MinMax_data(x1))
indata <- data1[,2:4]
outdata <- data1[,1]
pred <- predict(fit, indata)
View(pred)
plot(outdata, type = 'line', main='Résultat de modèle' )
lines(pred, col='red', lwd = 2)

round(cor(outdata, pred) ^ 2, 4)
round(cor(outdata, pred), 4)
MAE(outdata, pred)
RMSE(outdata, pred)

# regularzation # No solution founded to add the lambda parameter!!
## Dé-normaliser la série
unscale_data <- function (x)
{
  x * (max_data - min_data) + min_data
}
output_pred <- unscale_data (pred)
plot(x1, type = 'line')
lines(output_pred, col = "red", lwd=2)
legend(
  'topleft',
  # places a legend at the appropriate place
  c("sunspot", "Prévision"),
  # puts text in the legend
  # gives the legend appropriate symbols (lines)
  lwd = c(1.5, 2),
  col = c("black", "red")
)


#################
## Arima model ##
#################
#serie stationnaire x1

x1<- trainingSet
x1<- ts(x1)
plot(x1)
acf(x1)
pacf(x1)
y1=auto.arima(x1, ic = "aic", trace = T)
y1
tsdisplay(residuals(y1),lag.max = 20)
y2=Arima(x1,order = c(2,1,2))
y2
tsdisplay(residuals(y2),lag.max = 20)
adf.test(x1)
plot(x1)
fcast <- forecast(y2 , h=30)
plot(fcast)

############################
# Lissage avec Winters     #
############################
Holtfit <- HoltWinters(x1, )
plot(esfit)
