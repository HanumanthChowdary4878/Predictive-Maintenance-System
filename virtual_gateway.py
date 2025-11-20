#!/usr/bin/env python
# coding: utf-8

# In[3]:


import csv
from simulate_sensor import simulate_sensor

def start_virtual_gateway(filename="sensor_data.csv", samples=500):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Wind Speed", "Theoretical Power", "Wind Direction"])

        for _ in range(samples):
            data = simulate_sensor()
            writer.writerow([data["Wind Speed"], data["Theoretical Power"], data["Wind Direction"]])
    
    print("Virtual IoT Gateway Completed. Data stored in sensor_data.csv")

if __name__ == "__main__":
    start_virtual_gateway()


# In[ ]:




