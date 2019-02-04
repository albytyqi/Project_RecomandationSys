#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp,  removed) 
#we did not remove the MovieLens file since it will be used at the end 
#after the model is set to provide a full prediction table

#############################################################
# The Project
#############################################################

####Analysing the structure of the data#####
library(tidyverse)
print("dimension:")
dim(edx)
print("Structure:")
str(edx)

####rating distribution####
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) + 
  geom_line()+ ggtitle("rating distribution")

n_title <- n_distinct(edx$title)
n_movie <- n_distinct(edx$movieId)
n_user  <- n_distinct(edx$userId)

Data_structure_table <- data_frame(analyse="Distinct  Titles", total = n_title )
Data_structure_table <-bind_rows(Data_structure_table,data_frame(analyse="Distinct movie ", total = n_movie ))
Data_structure_table <-bind_rows(Data_structure_table,data_frame(analyse="Distinct User ", total = n_user ))

print("Data set table structure")
Data_structure_table %>% knitr::kable()

### list of different genres####
print("list of all movie genres")
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#rating distribution plot by movies####
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("rating distribution plot by movies")
#rating distribution plot by users ####
edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("rating distribution plot by users ")

#10 top rated movies####
print("10 top rated movies")
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

#10 top least rated movies####
print("10 top least rated movie")
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange((count))

#creating RMSE function####


RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#processing the first step of the Recomandation System by
#applying naively the mu_hat( average ratings of alll movies) and by this process
#the naive rmse####
mu_hat <- mean(edx$rating)
naive_rmse <- RMSE(validation$rating, mu_hat)

print("mu_hat")
mu_hat
print("naive_rmse")
naive_rmse

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
#this table will be used to evaluate the improvement of each model

rmse_results %>% knitr::kable()

#adding the movie effect to the naive model 
#naive rmse+movie effect which will be called moel_1_rmse and added to the table of RMSE

mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
#output: Table with RMSE
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()

#naive rmse+movie effect+user effect

user_avgs <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
model_2_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

#naive rmse+movie effect+user effect +genres effect
genres_avgs <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u +b_g) %>%
  .$pred
model_g_rmse <- RMSE(predicted_ratings, validation$rating)
#output: Table with RMSE####
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User  +genres Effects Model",  
                                     RMSE = model_g_rmse ))
rmse_results %>% knitr::kable()

#########################
##regularization model ####
#########################

####creating a sequence oflambda for evaluating best lambda as penalty factor

lambdas <- seq(0, 10, 0.25)

#regularizing by movie effect####
#creating function to assess best RMSE for movie effect with optimal Lambda
rmses_b_i <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    mutate(pred = mu + b_i ) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

#plotting the result of Lambda vs RMSE for better visualiation
qplot(lambdas, rmses_b_i)

#best lambda for movie effect model

lambda_i <- lambdas[which.min(rmses_b_i)]
lambda_i

#results incoporating to RMSE_results table
rmse_results <- bind_rows(rmse_results,
                            data_frame(method="Regularized Movie Effect Model",  
                                       RMSE = min(rmses_b_i)))
#printing the table of rmse results
rmse_results %>% knitr::kable()

#regularizing by user effect####
#creating function to assess best RMSE for user effect with optimal Lambda (movie effect already set to the model)

rmses_b_u <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda_i))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i+ b_u ) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses_b_u)  

lambda_u <- lambdas[which.min(rmses_b_u)]
print("lambda_u:")
lambda_u

#output: Table with RMSE

rmse_results <- bind_rows(rmse_results,
                            data_frame(method="Regularized Movie + User Effect Model",  
                                       RMSE = min(rmses_b_u)))
rmse_results %>% knitr::kable()

#regularizing by genre ####
#creating function to assess best RMSE for user effect with optimal Lambda (movie and user effect already set to the model)

rmses_b_g <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda_i))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_u))
  
  b_g <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId")%>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i- b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by= "genres") %>%
    mutate(pred=mu + b_i+b_u+b_g)  %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

#plotting the results of lambda_g vs RMSE by genre effect model
qplot(lambdas, rmses_b_g)  
#optimal lambda for genre effect model
lambda_g <- lambdas[which.min(rmses_b_g)]
print("lambda_g")
lambda_g

#table with final RMSE results including regularized Movie, user and genre effect model
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User + Genres Effect Model",  
                                     RMSE = min(rmses_b_g)))
rmse_results %>% knitr::kable()

# checking results #creating predicted ratings #

b_i <- validation %>%   group_by(movieId) %>%  summarize(b_i = sum(rating - mu)/(n()+lambda_i))

b_u <- validation %>%   left_join(b_i, by="movieId") %>%  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_u))

b_g <- validation %>%   left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId")%>%  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i- b_u - mu)/(n()+lambda_g))

predicted_ratings_rep <-   validation %>%   left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%  left_join(b_g, by = "genres") %>%
  mutate(pred=mu + b_i+b_u+b_g)

#check prediction in one movie#
library(lubridate)
shaw<-predicted_ratings_rep %>% filter(title=="Shawshank Redemption, The (1994)") %>%
  group_by(date=as_datetime(timestamp))%>%
  summarize(P=mean(pred),R=mean(rating),diff=R-P)%>% 
  arrange(date)

shaw%>%ggplot(aes(date,P),colour="red")+ geom_point()+ ggtitle("The Shawshank Redemption rating predictions")

########################
### regularizing by genre capped

rmses_b_g <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda_i))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_u))
  
  b_g <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId")%>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i- b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by= "genres") %>%
    mutate(pred=ifelse((mu + b_i+b_u+b_g)>5,5,mu + b_i+b_u+b_g) ) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

#plot showing lambda vs rmses_b_g
qplot(lambdas, rmses_b_g)  

#optimal lambda for capped genre effect model
lambda_g <- lambdas[which.min(rmses_b_g)]
print("lambda_g for capped model")
lambda_g

#final rmse result table with capped movie, user and genre Effect Model

rmse_results <- bind_rows(rmse_results,
                            data_frame(method="Regularized Movie + User + Genres Effect Model capped",  
                                       RMSE = min(rmses_b_g)))
rmse_results %>% knitr::kable()

#creating predicted ratings for whole movie table#
#finding final b_i,b_u,b_g

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda_i))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_u))

b_g <- edx %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId")%>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i- b_u - mu)/(n()+lambda_g))

#whole table with predicted ratings with 1M rows (predicted ratings=mu+b_i+b_u+b_g)####
predicted_ratings_rep <- 
  movielens %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred_c=ifelse((mu + b_i+b_u+b_g)>5,5,mu + b_i+b_u+b_g))

#whole table with predicted ratings for each movie with 10K rows####
predicted_ratings_rep_bymovie <- predicted_ratings_rep %>%group_by(title) %>%
summarize(n = n(), avg_ratings = mean(rating), pred_ratings = mean(pred_c)) %>%
  arrange(desc(n)) %>%
  mutate(title = reorder(title, avg_ratings)) 

#snapshot of 20 movie prediction####
print("snapshot of 20 movie prediction")
head(predicted_ratings_rep_bymovie,n=20L)
#snapshot of predicted rating for 20 first ratings by timestamp####
print("snapshot of predicted rating for 20 first ratings by timestamp")
predicted_ratings_rep%>% arrange(timestamp) %>% select(userId,movieId,rating, timestamp,title,genres,pred_c)%>%head(n=20L)
                                                                        