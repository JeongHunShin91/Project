
library(SyncRNG) 
library(nnet)
library(neuralnet)
library(ModelMetrics)
#데이터 불러오기&전처리
df<-read.csv("C:/Google driver/pythonProject_study/diabetes.csv")


#종속변수 factor형으로
df$Outcome<-as.factor(df$Outcome)

#데이터분할
v <- 1:nrow(df)
s <- SyncRNG(seed=42)
idx <- s$shuffle(v)[1:round(nrow(df)*0.7)] 
idx[1:length(idx)]
train <- df[idx,] 
test <- df[-idx,]


# 모델링
nn_model <- neuralnet(Outcome ~.,
                      data=train,
                      hidden=c(2),,
                      threshold=0.01,
                      stepmax = 1e+05,
                      rep = 1)
# threshold : 에러의 감소분이 threshold 값보다 작으면 stop
# hidden : hidden node 수. 
# hidden=c(2,2) : hidden layer 2개가 각각 hidden node 2개를 가짐
# linear.output: 활성함수('logistic' or 'tanh')가 출력 뉴런에 적용되지 않아야 하는 경우(즉, 회귀) TRUE로 설정(default)
# stepmax: 훈련 수행 최대 횟수



# 성능비교
nn_pred <- predict(nn_model, newdata=test)
nn_pred <- ifelse(nn_pred[,2] >= 0.5, 1, 0)


#평가
caret::confusionMatrix(as.factor(nn_pred),test$Outcome)$byClass
######## 
# 0.83

# ==============================================================================================================/
  

library(SyncRNG) 
library(nnet)
library(caret)
library(neuralnet)
library(ModelMetrics)
#데이터 불러오기&전처리
df<-read.csv("C:/Google driver/pythonProject_study/diabetes.csv")


#종속변수 factor형으로
# df$Outcome<-as.factor(df$Outcome)

#데이터분할
v <- 1:nrow(df)
s <- SyncRNG(seed=42)
idx <- s$shuffle(v)[1:round(nrow(df)*0.7)] 
idx[1:length(idx)]
train <- df[idx,] 
test <- df[-idx,]


# 모델링
nn_model <- neuralnet(Outcome ~.,
                      data=train,
                      hidden=c(2),,
                      threshold=0.01,
                      stepmax = 1e+05,
                      rep = 1)
# threshold : 에러의 감소분이 threshold 값보다 작으면 stop
# hidden : hidden node 수. 
# hidden=c(2,2) : hidden layer 2개가 각각 hidden node 2개를 가짐
# linear.output: 활성함수('logistic' or 'tanh')가 출력 뉴런에 적용되지 않아야 하는 경우(즉, 회귀) TRUE로 설정(default)
# stepmax: 훈련 수행 최대 횟수


# caret 기본모델
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 1)

caret_nn_model<- train(Outcome ~ .,
                       data = train,
                       method = "nnet" ,
                       trControl = fitControl)

caret_nn_pred<-predict(caret_nn_model, newdata = test)
caret_nn_pred <- ifelse(caret_nn_pred >= 0.5, 1, 0)


#평가
caret::confusionMatrix(as.factor(caret_nn_pred),as.factor(test$Outcome))$byClass

# ==================================================================================================
# caret tune 모델
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 1)

nn_Grid <-  expand.grid(size = 15,
                        decay = 2)

caret_tune_nn_model<- train(Outcome ~ .,
                            data = train,
                            method = "nnet" ,
                            tuneGrid = nn_Grid,
                            trControl = fitControl)

caret_tune_nn_pred<-predict(caret_tune_nn_model, newdata = test)
caret_tune_nn_pred <- ifelse(caret_tune_nn_pred >= 0.5, 1, 0)


#평가
caret::confusionMatrix(as.factor(caret_tune_nn_pred),as.factor(test$Outcome))$byClass





