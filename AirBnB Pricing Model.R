#Project 2
#Airbnb Pricing Preciditon from Neural Network modeling

library(caret) #For confusionMatrix(), training ML models, and more
library(neuralnet) #For neuralnet() function
library(dplyr) #For some data manipulation and ggplot
library(pROC)  #For ROC curve and estimating the area under the ROC curve
library(fastDummies) #To create dummy variable (one hot encoding)
library(sigmoid)

#Importing the data for Airbnb
airbnb = read.csv("AirbnbListings.csv", sep = ",", stringsAsFactors = TRUE)
str(airbnb)
summary(airbnb)

# Converting Superhost to numeric
airbnb$superhost = ifelse(airbnb$superhost, 1, 0)
# Checking for NA values in the entire dataset

colSums(is.na(airbnb)) / nrow(airbnb) * 100

# Output - 11% of host_acceptance_rate is missing. 
# 1% or less of data missing for room_type, total_reviews, avg_rating
# NOTE: Intuitively, host-acceptance rate should not impact the price they are able to charge. 
#       I will omit this data going forward prior to any models to ensure accuracy of predictions
# I will simply remove the other data since it is <1% of total data. 

airbnb = airbnb %>% select(-c(host_acceptance_rate))
airbnb_without_na <- airbnb[complete.cases(airbnb), ]

#Checking for NA again (none exists)
colSums(is.na(airbnb_without_na))

#Dates are loaded as factors, converting to date format then Unix reference date 
airbnb_without_na$host_since = as.numeric(format(as.Date(airbnb_without_na$host_since, format = "%m/%d/%Y"),"%Y"))
airbnb_without_na$host_years = as.numeric(2023 - airbnb_without_na$host_since)


#Create dummy variables for each level of the categorical variables of interest
airbnb_dummies = dummy_cols(airbnb_without_na, select_columns = c('neighborhood','room_type'))

#Removing Listing_ID (irrelvant) 
#Also, removing one dummy level of each categorical variables

final_data = 
  airbnb_dummies %>% select(-c(listing_id, neighborhood,  host_since, room_type, `room_type_EntireHome/apt`, neighborhood_Bellevue))

#Classification methods are applied to categorical variables
#Changing the data type (of the target variable, Survived) to factor:
str(final_data)

#Data splitting into training and test (70%:30%)
set.seed(123)  
index = sample(nrow(final_data),0.7*nrow(final_data)) 

train_data = final_data[index, ]
test_data = final_data[-index, ]

# Using a min-max normalization for all values
scale_vals = preProcess(train_data, method="range")
train_data_s = predict(scale_vals, train_data)
test_data_s = predict(scale_vals, test_data)


#Neural Network Model

# Starting with a simple model
# Linear Activiation (ensures continuous unbounded response)
# Stepmax set to did not converge (1000, 10000)
# Activation function: Rectified Linear
# Hidden: 5

#Model 1:
NN1 = neuralnet(price~.,
                data=train_data_s,
                linear.output = TRUE,
                stepmax = 1e+05,
                act.fct = 'logistic')

#The output model:
NN1
plot(NN1)

#predicted values for test data (these will be between 0 and 1)
pred1 = predict(NN1, test_data_s)

#Scaling back predicted values to the actual scale of price
pred1_acts = pred1*(max(train_data$price)-min(train_data$price))+min(train_data$price)

plot(test_data$price,pred1_acts, xlab="Price",ylab="Predicted Price",main="Model 1 (Logistic)")

postResample(pred1_acts,test_data$price) #Model 1

#Model 1b:
NN1b = neuralnet(price~.,
                data=train_data_s,
                linear.output = TRUE,
                stepmax = 1e+05,
                act.fct = 'logistic',
                hidden = c(2,3))

#The output model:
NN1b
plot(NN1b)

#predicted values for test data (these will be between 0 and 1)
pred1b = predict(NN1b, test_data_s)

#Scaling back predicted values to the actual scale of price
pred1b_acts = pred1b*(max(train_data$price)-min(train_data$price))+min(train_data$price)

plot(test_data$price,pred1b_acts, xlab="Price",ylab="Predicted Price",main="Model 1 (Logistic w/ Hidden)")

postResample(pred1b_acts,test_data$price) #Model 1b


#Model 2
ctrl = trainControl(method="cv",number=10)
myGrid = expand.grid(size = seq(1,10,1),
                     decay = seq(0.01,0.2,0.04))

set.seed(123)
NN2 = train(
  price ~ ., data = train_data_s,
  linout = TRUE,
  method = "nnet", 
  tuneGrid = myGrid,
  trControl = ctrl,
  trace=FALSE)

#The output model:
NN2

#predicted values for test data (these will be between 0 and 1)
pred2 = predict(NN2, test_data_s)

#Scaling back predicted values to the actual scale of price
pred2_acts = pred2*(max(train_data$price)-min(train_data$price))+min(train_data$price)

plot(test_data$price,pred2_acts,xlab="Price",ylab="Predicted Price",main="Model 2 (Caret)")

#Model 3:
NN3 = neuralnet(price~.,
                data=train_data_s,
                linear.output = TRUE,
                stepmax = 1e+05,
                act.fct = 'tanh',
                hidden= c(2,3))

#The output model:
NN3

#predicted values for test data (these will be between 0 and 1)
pred3 = predict(NN3, test_data_s)

#Scaling back predicted values to the actual scale of price
pred3_acts = pred3*(max(train_data$price)-min(train_data$price))+min(train_data$price)

plot(test_data$price,pred3_acts, xlab="Price",ylab="Predicted Price",main="Model 3 (Tangent)")



#Models comparison
postResample(pred1_acts,test_data$price) #Model 1 Logistic
postResample(pred1b_acts,test_data$price) #Model 1b Logistic (hidden 2,3)
postResample(pred2_acts,test_data$price) #Model 2 Caret
postResample(pred3_acts,test_data$price) #Model 3 Tangent (hidden 2,3)

# Plots for all NN models
plot(NN1)
plot(NN1b)
plot(NN2)
plot(NN3)
