
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

###########모델 train############
#SVM 모델
help(lm)
help(svm)
library(e1071)
svm_model<-svm(Outcome~.,train_df)
svm_pred<-predict(svm_model,subset(test_df,select=-c(Outcome)))

#caret모델
caret_svm_model <- train(Outcome ~ ., data = train_df, 
                 method = "svmLinearWeights",tuneLength=5)
caretsvm_pred<-predict(caret_svm_model, newdata = test_df)

#caret 튜닝모델
caretsvmGrid <-  expand.grid(cost= 5,
                             weight = 1)
caret_tune_svm_model <- train(Outcome ~ ., data = train_df, 
                         method = "svmLinearWeights",
                         tuneGrid= caretsvmGrid
                        )
caret_tune_svm_pred<-predict(caret_tune_svm_model, newdata = test_df)

#######################################
#Adaboost 모델
library(remotes)
install_github("cran/fastAdaboost")
# install.packages('adabag') 
library(fastAdaboost)
library(adabag)
ada_model<-boosting(Outcome~.,train_df)
ada_pred<-predict(ada_model,subset(test_df,select=-c(Outcome)))

#caret모델
a<-expand.grid(nIter=200)
caret_ada_model <- train(Outcome ~ ., data = train_df, method = 'adaboost',tuneGride=a)
caretada_pred<-predict(caret_ada_model, newdata = test_df)

#R 튜닝모델
R_tune_ada_model <- boosting(Outcome~.,train_df,mfinal = 10, coeflearn = "Zhu")
R_tune_ada_pred<-predict(R_tune_ada_model,subset(test_df,select=-c(Outcome)))

#######################################
#로지스틱 모델
help(glm)
logit_model<-glm(Outcome ~., family=binomial(), data=train_df)
logit_pred <- predict(logit_model, newdata = test_df, type = 'response')
logit_pred <-ifelse(logit_pred>0.5,1,0)

#caret모델
caret_logit_model <- train(Outcome ~ ., data = train_df, method = 'LogitBoost',nIter=10)
caretlogit_pred<-predict(caret_logit_model, newdata = test_df)

#######################################


#svm평가
caret::confusionMatrix(svm_pred,test_df$Outcome)$byClass #svm 평가
caret::confusionMatrix(caretsvm_pred,test_df$Outcome)$byClass #caret평가
caret::confusionMatrix(caret_tune_svm_pred,test_df$Outcome)$byClass #caret tune평가

#ada평가
caret::confusionMatrix(as.factor(ada_pred$class),test_df$Outcome)$byClass #adabooost 평가
caret::confusionMatrix(as.factor(R_tune_ada_pred$class),test_df$Outcome)$byClass #R tune평가
caret::confusionMatrix(caretada_pred,test_df$Outcome)$byClass #caret평가

#로지스틱 평가
caret::confusionMatrix(as.factor(logit_pred),test_df$Outcome)$byClass[7] #로지스틱 평가
caret::confusionMatrix(as.factor(caretlogit_pred),test_df$Outcome)$byClass #caret 평가
caret::confusionMatrix(as.factor(logit_pred),test_df$Outcome)$overall #로지스틱 평가

