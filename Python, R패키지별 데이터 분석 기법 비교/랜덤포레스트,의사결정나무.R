# R Baseline
library(SyncRNG) 
library(caret)
library(ModelMetrics)
#데이터 불러오기&전처리
df<-read.csv('diabetes.csv')

#종속변수 factor형으로
df$Outcome<-as.factor(df$Outcome)

#데이터분할
v <- 1:nrow(df)
s <- SyncRNG(seed=42)
idx <- s$shuffle(v)[1:round(nrow(df)*0.7)] 
idx[1:length(idx)]
train_df <- df[idx,] 
test_df <- df[-idx,]

###########랜덤포레스트 모델 train############
#R 모델
library(randomForest)
random_model<-randomForest(Outcome~.,train_df)
random_pred<-predict(random_model,subset(test_df,select=-c(Outcome)))

#caret 기본 모델
caret_random_model <- train(Outcome ~ ., data = train_df, 
                         method = "rf")
caretrandom_pred<-predict(caret_random_model, newdata = test_df)

#caret 튜닝모델
mtry <- sqrt(ncol(train_df))
caretrandomGrid <-  expand.grid(mtry = mtry)
caret_tune_random_model <- train(Outcome ~ ., data = train_df, 
                              method = "rf",
                              tuneGrid= caretrandomGrid)
caret_tune_random_pred<-predict(caret_tune_random_model, newdata = test_df)
################################

#평가
caret::confusionMatrix(random_pred,test_df$Outcome)$byClass #R 평가
caret::confusionMatrix(caretrandom_pred,test_df$Outcome)$byClass #caret평가
caret::confusionMatrix(caret_tune_random_pred,test_df$Outcome)$byClass #caret tune평가

###########의사결정나무 모델 train############
#R 모델
# install.packages("tree")
library(rpart)
tree_model<- rpart(Outcome~., data= train_df)
tree_pred<-predict(tree_model,subset(test_df,select=-c(Outcome)),type='class')
#caret 기본 모델
# fit the model
caret_tree_model = train(Outcome ~ ., 
                  data=train_df, 
                  method="rpart", 
                  )
caret_tree_pred<-predict(caret_tree_model, newdata = test_df)
#caret 튜닝모델
mtry <- sqrt(ncol(train_df))
cp_grid <- data.frame(cp = seq(0.02, .2, .02))
caret_tune_tree_model <- train(Outcome ~ ., data = train_df, 
                                 method = "rpart",
                               trControl=trainControl(method='cv'))
caret_tune_tree_pred<-predict(caret_tune_tree_model, newdata = test_df)
################################

#평가
library(e1071)
treepred <- predict(tree_model,test_df,type="class")
plot(treepred)
confusionMatrix(treepred,test_df$Outcome)

caret::confusionMatrix(tree_pred,test_df$Outcome)$byClass #R 평가
caret::confusionMatrix(caret_tree_pred,test_df$Outcome)$byClass #caret평가
caret::confusionMatrix(caret_tune_tree_pred,test_df$Outcome)$byClass #caret tune평가
