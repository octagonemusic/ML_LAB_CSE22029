import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "lab02/Lab Session Data.xlsx"
sheet_name = "IRCTC Stock Price"
data = pd.read_excel(file_path, sheet_name=sheet_name)

price = data["Price"]

#POPULATION
population_mean = np.mean(price)
population_variance = np.var (price)

print(f"The population mean price is: {population_mean:.2f}")
print(f"The population variance price is: {population_variance:.2f}")

#WEDNESDAYS
wednesday_data = data[data["Day"] == "Wed"]
wednesday_price = wednesday_data["Price"]
wednesday_mean = np.mean(wednesday_price)

print(f"\nThe mean price on wednesdays is: {wednesday_mean:.2f}")       

#APRIL
april_data = data[data["Month"] == "Apr"]
april_price = april_data["Price"]
april_mean = np.mean(april_price)

print(f"\nThe mean price in April is: {april_mean:.2f}")

#PROBABILITY OF PROFIT/LOSS
chg_percent = data["Chg%"]
loss_days = chg_percent[chg_percent < 0].count()
total_days = chg_percent.count()
profit_days = total_days - loss_days
probability_of_loss = (loss_days / total_days) * 100

print(f"\nThe probability of making a loss over this stock is: {probability_of_loss:.3f}%")

#PROBABILITY OF PROFIT ON WEDNESDAY
wed_chg_percent = wednesday_data["Chg%"]
wed_profit_days = wed_chg_percent[wed_chg_percent > 0].count()
wed_total_days = wed_chg_percent.count()
wed_profit_probability = (wed_profit_days / wed_total_days) * 100

print(f"\nThe probability of making a profit over this stock on wednesdays is: {wed_profit_probability:.3f}%")

#CONDITIONAL PROBABILITY OF PROFIT GIVEN THAT IT'S WEDNESDAY
wed_probability = wed_total_days / total_days
conditional_profit_probability = (wed_profit_probability / wed_total_days) * 100

print(f"\nThe conditional probability of making a profit given that today is Wednesday is: {conditional_profit_probability:.3f}%")

#SCATTER PLOT OF Chg% DATA AGAINST DAY OF WEEK
days = data["Day"]

day_mapping = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
days_numeric = days.map(day_mapping)

# Create the scatter plot
chg_percent_100 = chg_percent * 100
plt.figure(figsize=(10, 6))
plt.scatter(days_numeric, chg_percent_100, alpha=0.6)
plt.xticks(ticks=list(day_mapping.values()), labels=list(day_mapping.keys()))
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Scatter Plot of Chg% Against Day of the Week")
plt.grid(True)
plt.show()  