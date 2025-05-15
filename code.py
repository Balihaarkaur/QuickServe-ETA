# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("deliverytime.txt")
print(data.head())

# Define Earth's radius in kilometers
R = 6371

# Function to convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi / 180)

# Function to calculate distance using the Haversine formula
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate distance for each delivery
data['distance'] = data.apply(lambda row: distcalculate(row['Restaurant_latitude'],
                                                        row['Restaurant_longitude'],
                                                        row['Delivery_location_latitude'],
                                                        row['Delivery_location_longitude']), axis=1)

# Exploratory Data Analysis
# Relationship between Delivery Person Age and Time Taken
fig_age = px.scatter(data_frame=data,
                     x="Delivery_person_Age",
                     y="Time_taken(min)",
                     size="Time_taken(min)",
                     color="distance",
                     trendline="ols",
                     title="Relationship Between Time Taken and Age")
fig_age.show()

# Relationship between Delivery Person Ratings and Time Taken
fig_ratings = px.scatter(data_frame=data,
                         x="Delivery_person_Ratings",
                         y="Time_taken(min)",
                         size="Time_taken(min)",
                         color="distance",
                         trendline="ols",
                         title="Relationship Between Time Taken and Ratings")
fig_ratings.show()

# Box plot for Type of Vehicle and Type of Order
fig_vehicle_order = px.box(data,
                           x="Type_of_vehicle",
                           y="Time_taken(min)",
                           color="Type_of_order",
                           title="Time Taken by Vehicle Type and Order Type")
fig_vehicle_order.show()

# Prepare data for modeling
features = data[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]]
target = data["Time_taken(min)"]

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.10, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Predict delivery time for new input
print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = float(input("Total Distance (in km): "))

# Create a DataFrame for the new input
new_data = pd.DataFrame([[a, b, c]], columns=["Delivery_person_Age", "Delivery_person_Ratings", "distance"])

# Predict and display the delivery time
predicted_time = model.predict(new_data)
print("Predicted Delivery Time in Minutes = ", predicted_time[0])
