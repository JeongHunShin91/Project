# 1. 제품적절성이 제품만족도에 미치는 영향 주제로 R을 이용한 
# 단순 선형 회귀분석을 실시하고, 딥러닝 교재 3장에서 사용된 
# 인공신경망을 이용한 선형 회귀분석을 python으로 실행하여 
# 결과를 비교하시오.
product <- read.csv("dataset4/product.csv", header = TRUE)
str(product)

y = product$제품_만족도
x = product$제품_적절성
df <- data.frame(x, y)

lm <- lm(formula = y ~ x, data = product)
summary(lm)

# 산점도 그래프 그리기
plot(formula = y ~ x, data = product, 
     xlab ="제품_적절성", ylab = "제품_만족도" ,col = 1, cex = 2)

# 회귀선 그래프 그리기
abline(lm, col = "red")

# 텍스트 삽입
text(1.3,5, labels="R2 = 58.65%")
text(1.47,4.8, labels="y=0.7392x +0.7788")


# 2. 비 유무 예측 주제로 R을 이용한 로지스틱 회귀분석을 
# 실시하고, 딥러닝 교재4장에서 사용된 인공신경망을 이용한 
# 로지스틱 회귀분석을 python으로 실행하여 결과를 비교하시오.

library(car)
library(lmtest)
library(ROCR)
library(SyncRNG)

weather = read.csv("dataset4/weather.csv", stringsAsFactors = F)
head(weather, 3)

weather_df <- weather[ , c(-1, -6, -8, -14)]
weather_df$RainTomorrow[weather_df$RainTomorrow == 'Yes'] <- 1
weather_df$RainTomorrow[weather_df$RainTomorrow == 'No'] <- 0
weather_df$RainTomorrow <- as.numeric(weather_df$RainTomorrow)
head(weather_df, 3)

v <- 1:nrow(weather_df)
s <- SyncRNG(seed=42)
s=s$shuffle(v)
idx <- s[1:round(nrow(weather_df)*0.7)]

head(weather_df[-idx[1:length(idx)],])
idx[1:length(idx)]
train <- weather_df[idx[1:length(idx)],]
test <- weather_df[-idx[1:length(idx)],]

weather_model <- glm(RainTomorrow ~ ., data = train, family = 'binomial', na.action=na.omit)
summary(weather_model)

pred <- predict(weather_model, newdata = test, type = "response")

result_pred <- ifelse(pred >= 0.5, 1, 0)

table(result_pred, test$RainTomorrow)
sum(result_pred == test$RainTomorrow) / length(test$RainTomorrow)
# 정확도 :  0.8727273

# ROC curve 시각화
pr <- prediction(pred, test$RainTomorrow)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)


# 3. 다중분류를 위한 머신러닝 기법을 선택하고 R을 이용하여 
# 다중분류를 실시하고, 딥러닝 교재 7장에서 사용된 텐서플로와 
# 케라스을 이용한 인공신경망을 python으로실행하여 결과를 
# 비교하시오.
library(randomForest)
data(iris)
iris
head(iris, 3)

# 70% training데이터, 30% testing데이터로 구분
set.seed(42)
idx <- sample(2, nrow(iris), replace=T, prob=c(0.7, 0.3))
trData <- iris[idx == 1, ]
nrow(trData)
teData <- iris[idx == 2, ]
nrow(teData)

# 랜덤포레스트 실행 (100개의 tree를 다양한 방법(proximity=T)으로 생성)
RFmodel <- randomForest(Species~., data=trData, ntree=100, proximity=T)
RFmodel

# 오차율 시각화
x11()
plot(RFmodel, main="RandomForest Model of iris")
legend("topright", colnames(RFmodel$err.rate),col=1:4,cex=1.5,fill=1:4)

# 테스트데이터로 예측
pred <- predict(RFmodel, newdata=teData)

# 실제값과 예측값 비교
table(teData$Species, pred)
sum(teData$Species == pred) / length(pred)

# 정확도 시각화
COLS = c("#ED0000FF","#00468BFF","#42B540FF")
names(COLS) =unique(teData$Species)
MAR = sort(margin(RFmodel,teData$Species))
plot(as.numeric(MAR),col=COLS[names(MAR)],pch=20,ylab="margin")
legend("bottomright",fill=COLS,names(COLS), cex=1.5)

