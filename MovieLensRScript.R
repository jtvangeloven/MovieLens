#Importing, cleaning, and organizing the data using the tidyverse and caret packages
if(!require(tidyverse)) install.packages("tiduverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl)) download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"),simplify = TRUE),
                         stringsAsFactors = FALSE)

colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>% 
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1, sample.kind = "Rounding")

#Creating the edx data set and the final holdout test set
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#I examined the data in the edx data set to determine a good starting point for building a model

#The number of individual ratings is given by
nrow(edx)

#The number of columns (variables) is given by
ncol(edx)

#Visualize how the data set is organized
head(edx)

#The number of movies rated is
length(unique(edx$movieId))

#The number of users that rated movies is
length(unique(edx$userId))  

#A visualization of the most common ratings 
edx %>% group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(rating, count)) +
  geom_line() +
  geom_point()

#The most common ratings are
common_rating <- edx %>%
  group_by(rating) %>%
  summarize(num_ratings = n()) %>%
  arrange(desc(num_ratings)) %>%
  top_n(10, num_ratings)
common_rating

#The distribution of ratings among movies is given by
edx %>% group_by(movieId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "white") +
  scale_x_log10() +
  ggtitle("Number of movies v. number of ratings per movie") +
  xlab("Number of ratings") +
  ylab("Number of movies")

#The amount of movies that each user rated is given by
edx %>% group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "white") +
  scale_x_log10() +
  ggtitle("Number of users v. number of ratings per user") +
  xlab("Number of ratings") +
  ylab("Number of users")

#The top 10 movies with the most ratings is
num_ratings <- edx %>% 
  group_by(movieId) %>%
  summarize(num_ratings = n(), movieTitle = first(title)) %>%
  arrange(desc(num_ratings)) %>%
  top_n(10, num_ratings)
num_ratings

#The most highly rated movies are
best_rating <- edx %>%
  group_by(movieId) %>%
  filter(mean(rating) == 5) %>%
  select(movieId, rating, title)
head(best_rating)


#The edx data set is split into a train set and a test set
indexes <- split(1:nrow(edx), edx$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) %>%
  unlist(use.names = TRUE) %>%
  sort()
test_set <- edx[test_ind,]
train_set <- edx[-test_ind,]

#Make sure the same movies are in the train set and the test set
test_set <- test_set %>%
  semi_join(train_set, by = "movieId")
train_set <- train_set %>%
  semi_join(test_set, by = "movieId")

#A function is created to calculate the RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#A random sample is created using the distribution of the ratings
p <- function(x, y) mean(y == x)
rating <- seq(0.5, 5, 0.5)

B <- 10000
M <- replicate(B, {
  s <- sample(train_set$rating, 100, replace = TRUE)
  sapply(rating, p, y = s)
})
prob <- sapply(1:nrow(M), function(x) mean(M[x,]))

y_hat_random <- sample(rating, size = nrow(test_set),
                       replace = TRUE, prob = prob)
result_random <- RMSE(test_set$rating, y_hat_random)
result_random


#Now the overall mean is used to create a model
#The average rating is:
mu <- mean(train_set$rating)
mu

#The resulting RMSE of the mean rating model is
result_mu <- RMSE(test_set$rating, mu)
result_mu


#Now we try the movie effect plus the mean model
#The movie effect is bi
bi <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
head(bi)

#Add the bi to the mean model
y_hat_bi <- mu + test_set %>%
  left_join(bi, by = "movieId") %>%
  .$b_i

#Calculate the RMSE of the mean + movie effect model
result_mu_bi <- RMSE(test_set$rating, y_hat_bi)
result_mu_bi


#Now we add the user effect to the movie effect and the mean model
#The user effect
bu <- train_set %>%
  left_join(bi, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#Add the bu to the model
y_hat_bi_bu <- test_set %>%
  left_join(bi, by = "movieId") %>%
  left_join(bu, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#Calculate the RMSE of the user effect + the movie effect + the mean model
result_mu_bi_bu <- RMSE(test_set$rating, y_hat_bi_bu)
result_mu_bi_bu


#Now we try regularization to improve the RMSE
#First we create a function to try out lambda values
regularized <- function(lambda, trainset, testset){
  mu <- mean(trainset$rating)
  b_i <- trainset %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + lambda))
  b_u <- trainset %>%
    left_join(b_i, by = "movieId") %>%
    filter(!is.na(b_i)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))
  predicted_ratings <- testset %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, testset$rating))
}

#Use a set of lambdas to tune
lambdas <- seq(0, 10, 0.25)

#Find the most accurate lambda
rmses <- sapply(lambdas,
                regularized,
                trainset = train_set,
                testset = test_set)

#Make a graph of the lambdas versus the rmses to visualize the best lambda
tibble(Lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point()

#We determine the best lambda
lambda <- lambdas[which.min(rmses)]
lambda

#Now we use regularization (the lambda) to reevaluate each model
#The mean
mu <- mean(train_set$rating)

#The regularized movie effect (the bi)
reg_bi <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + lambda))

#The regularized user effect (the bu)
reg_bu <- train_set %>%
  left_join(reg_bi, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))

# The prediction with the regularized mean, user and movie effect\
y_hat_reg <- test_set %>%
  left_join(reg_bi, by = "movieId") %>%
  left_join(reg_bu, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#Calculate the RMSE of the regularized user effect + the regularized movie effect +
# the mean model is
result_reg <- RMSE(test_set$rating, y_hat_reg)
result_reg


#We finally try our method on the final holdout test
#The final mean
mu_final <- mean(edx$rating)

#The final movie effect
bi_final <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + lambda))

#The final user effect
bu_final <- edx %>%
  left_join(bi_final, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + lambda))

#The final regularized prediction
final_y_hat_reg <- final_holdout_test %>%
  left_join(bi_final, by = "movieId") %>%
  left_join(bu_final, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#The final RMSE using the final holdout test is
final_result <- RMSE(final_holdout_test$rating, final_y_hat_reg)
final_result


