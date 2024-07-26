library(readxl)
library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)
library(ggplot2)
library(caret)
library(randomForest)
library(xgboost)
library(h2o)
library(readr)
library(cluster)

# Load data, fix dates. Update the column names.
data <- read_csv("C:/file.csv", 
                 col_types = cols(`MCVID (v29) (evar29)` = col_character(), 
                                  `24H Clock by Minute` = col_time(format = "%H:%M")))
data$Date <- as.Date(mdy(data$Date), format("%Y-%m-%d"))
data$timestamp <- with(data, as.POSIXct(paste(data$Date, data$`24H Clock by Minute`), format="%Y-%m-%d %H:%M:%S"))
#data_backup <- data

data <- data |> 
  rename(date = Date) |> 
  rename(pagename = `Page Name (v26) (evar26)`) |> 
  rename(userid = `MCVID (v29) (evar29)`) |> 
  rename(errordata = `Error Data (v32) (evar32)`) |> 
  rename(time = `24H Clock by Minute`) |> 
  rename(previousPage = `Previous Page (v99) (evar99)`) |> 
  rename(url = `URL (v8) (evar8)`) |> 
  rename(urlhash = `URL Hash (v39) (evar39)`) |> 
  rename(error_other = `Error - Other (ev27) (event27)`) |> 
  rename(error_server = `Error - Server (ev29) (event29)`) |> 
  rename(error_user = `Error - User (ev28) (event28)`) |> 
  rename(signin = `My Account: Sign In (event355)`) |> 
  rename(reg_step_1 = `My Account: Registration Step 1.0 (s) (event350)`) |> 
  rename(reg_step_2 = `My Account: Registration Step 2.0 (s) (event351)`) |> 
  rename(reg_step_3 = `My Account: Registration Step 3.0 (s) (event352)`) |> 
  rename(pageviews = `Page Views`) |> 
  rename(visit_num = `Visit Number`) |> 
  rename(visits = Visits) |> 
  rename(uniquevisitors = `Unique Visitors`) |> 
  drop_na(userid) |> 
  mutate(errordata = ifelse(is.na(errordata), "None", errordata)) |> 
  mutate(pagename = ifelse(is.na(pagename), "None", pagename)) |> 
  mutate(urlhash  = ifelse(is.na(urlhash), "None", urlhash)) |> 
  mutate(previousPage = ifelse(is.na(previousPage), "None", previousPage)) |> 
  relocate(timestamp, .before = userid) |> 
  select(!(`Visit Number (v15) (evar15)`))

data <- na.omit(data)
# View(data)
# Inspect the structure of the dataset
# str(data)

total_users <- data |> 
  group_by(userid) |> 
  summarise(n_distinct(userid))

# Feature engineering
user_session_features <- data %>%
  group_by(userid, visit_num) %>%
  summarise(total_errors = sum(error_other + error_server, na.rm = TRUE),
            session_length_hours = as.numeric(difftime(max(timestamp), min(timestamp), units = "hours")),
            session_length_mins = as.numeric(difftime(max(timestamp), min(timestamp), units = "mins")),
            session_length_days = as.numeric(difftime(max(timestamp), min(timestamp), units = "days")),
            session_length_weeks = as.numeric(difftime(max(timestamp), min(timestamp), units = "weeks")),
            error_type = n_distinct(errordata),
            error_count = cumsum(total_errors),
            errors_last_24h = sum(ifelse(is.na(total_errors[timestamp >= max(timestamp) - 86400]), 0, total_errors[timestamp >= max(timestamp) - 86400])),
           # time_since_last_error = as.numeric(difftime(timestamp, max(timestamp[error_occurred == 1]), units = "hours")),
            avg_sign_in = sum(signin, na.rm = TRUE),
            total_pages = n_distinct(pagename)) |> 
  ungroup()

# user_counts <- user_session_features |> 
#   group_by(userid) |> 
#   summarise(cluster)

# Scale the data for clustering
user_session_scaled <- user_session_features %>%
  select(-userid) %>%
  scale()

# Determine the optimal number of clusters using the Elbow Method

wss <- (nrow(user_session_scaled)-1)*sum(apply(user_session_scaled, 2, var))
for (i in 2:15) wss[i] <- sum(kmeans(user_session_scaled, centers=i)$withinss)

# Plot the Elbow Method
plot(1:15, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Apply k-means clustering with the chosen number of clusters
set.seed(123)
k <- 5 # Assuming 4 clusters based on the Elbow Method plot
kmeans_result <- kmeans(user_session_scaled, centers = k, nstart = 25)

# Add cluster results to the original data
user_session_features$cluster <- kmeans_result$cluster

# Analyze clusters
cluster_summary <- user_session_features %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)))

# Visualize clusters
ggplot(user_session_features, aes(x = total_errors, y = session_length_hours, color = factor(cluster))) +
  geom_point() +
  labs(title = "K-means Clustering of User Sessions", x = "Total Errors", y = "Session Length (Hours)", color = "Cluster")

# Output cluster summary
View(cluster_summary)

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(user_session_features$total_errors, p = .8, list = FALSE, times = 1)
trainData <- user_session_features[trainIndex,]
testData <- user_session_features[-trainIndex,]

# Build and evaluate Random Forest model
rf_model <- randomForest(total_errors ~ ., data = trainData, importance = TRUE)
predictions <- predict(rf_model, newdata = testData)
confusionMatrix(predictions, testData$total_errors)

# Visualization (optional)
ggplot(user_session_features, aes(x = total_errors)) +
  geom_histogram(binwidth = 1) +
  labs(title = "Distribution of Total Errors per User Session")

