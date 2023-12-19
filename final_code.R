
# load data
data <- read.csv('College.csv')
# (a) (5 points) Split the data set into a training set and a test set.

train <- sample(1:nrow(x), 0.5*nrow(x), replace = FALSE)  # half training, half test
x.test <- x[-train,]
y.test <- y[-train]

# (b) (5 points) Fit a ridge regression model on the training set, with λ chosen by cross-validation. Report the test error obtained.

set.seed(123)
cv.out.ridge <- cv.glmnet(x[train,], y[train], alpha=0)## use k-fold CV to choose the best lambda
plot(cv.out.ridge)
bestlam.ridge <- cv.out.ridge$lambda.min
bestlam.ridge #the best lambda is 0.09488765

ridge.mod <- glmnet(x[train,], y[train], alpha=0, lambda=grid)
ridge.pred <- predict(ridge.mod, s=bestlam.ridge, newx = x.test) 
mean((ridge.pred - y.test)^2)  # test MSE is 0.02341548

ridge.out <- glmnet(x, y, alpha=0)  # whole data, default grid
ridge.out.pred <- predict(ridge.out, type="coefficients", s=bestlam.ridge)[1:24,]
ridge.out.pred
ridge.out.pred[order(unlist(ridge.out.pred))] 

# (c) (5 points) Fit a lasso model on the training set, with λ chosen by cross-validation. Report the test error obtained, along with the number of non-zero coefficient estimates.
set.seed(123)
cv.out.lasso <- cv.glmnet(x[train,], y[train], alpha=1)
plot(cv.out.lasso)
bestlam.lasso <- cv.out.lasso$lambda.min
bestlam.lasso #the best lambda is 0.01197346

lasso.mod <- glmnet(x[train,], y[train], alpha=1, lambda=grid)
lasso.pred <- predict(lasso.mod, s=bestlam.lasso, newx = x.test)
mean((lasso.pred - y.test)^2)# test MSE is 0.009198251

lasso.out <- glmnet(x, y, alpha=1)# whole data, default grid
lasso.pred.out <- predict(lasso.out, type="coefficients", s=bestlam.lasso)[1:24, ]
lasso.pred.out  # a few coefficients are exactly zero
lasso.pred.out[lasso.pred.out!=0]  
# (d) (5 points) Pick one or two focal variables and run a double-lasso regression on the whole dataset. Briefly comment on the coefficients you get from the three models above.
