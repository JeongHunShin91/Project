# install.packages("xgboost")
# install.packages("Matrix")
library(xgboost)
library(Matrix)

df<-read.csv("C:/Google driver/pythonProject_study/diabetes.csv")
df_xgboost <- df #원본데이터를 변경하지 않기 위해 복사
df

df_xgb_sparse_matrix <- sparse.model.matrix(Outcome~.-1,data=df_xgboost)

# train, test data set, label data 생성
library(SyncRNG) 
v <- 1:nrow(df)
s <- SyncRNG(seed=42)
idx <- s$shuffle(v)[1:round(nrow(df_xgb_sparse_matrix)*0.7)] 
idx[1:length(idx)]
# train <- df[idx,] 
# test <- df[-idx,]
train_x <- df_xgb_sparse_matrix[idx,]
test_x <- df_xgb_sparse_matrix[-idx,]
train_y <- df_xgboost[idx,'Outcome']
test_y <- df_xgboost[-idx,'Outcome']

# xgboost알고리즘 사용을 위한 데이터 형태 변환
dtrain <- xgb.DMatrix(data = train_x, label = as.matrix(train_y))
dtest <- xgb.DMatrix(data = test_x, label = as.matrix(test_y))



# xgb모델 생성
xgbmodel <- xgboost(data = dtrain,                     
                    max.depth=5,                            
                    nrounds=20)                             

# xgb모델 예측
XGB_pred <- predict(xgbmodel, dtest)
XGB_pred <- ifelse(XGB_pred >= 0.5, 1, 0)

# 해당 모델의 중요 변수
library(dplyr)
xgb.importance(colnames(dtrain), model = xgb) %>% 
  xgb.plot.importance(top_n = 30)


# 해당 모델 저장
xgb.save(xgb,"xgboost.model")



# caret 기본모델
caret_xgb_model<- train(Outcome ~ ., data = train,
                        method = "xgbTree", tuneGrid= caretxgbGrid)
caret_xgb_pred<-predict(caret_xgb_model, newdata = test)

# caret 튜닝모델
caretxgbGrid <-  expand.grid(nrounds = 200,
                             max_depth = 5,
                             eta = 0.05,
                             gamma = 0.01,
                             colsample_bytree = 0.75,
                             min_child_weight = 0,
                             subsample = 0.5)

caret_tune_xgb_model <- train(Outcome ~ ., data = train, 
                              method = "xgbTree", tuneGrid= caretxgbGrid)

caret_tune_xgb_pred<-predict(caret_tune_xgb_model, newdata = test)


# f1 score
caret::confusionMatrix(as.factor(test_y), as.factor(XGB_pred))$byClass #R 평가  #0.8407643
caret::confusionMatrix(caret_xgb_pred,test$Outcome)$byClass #caret평가 # 0.8246753
caret::confusionMatrix(caret_tune_xgb_pred,test$Outcome)$byClass #caret tune평가 #0.8424437
