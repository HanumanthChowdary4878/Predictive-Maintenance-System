#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

def simulate_sensor():
    data = {
        "Wind Speed": round(random.uniform(3, 25), 2),
        "Theoretical Power": round(random.uniform(100, 5000), 2),
        "Wind Direction": round(random.uniform(0, 360), 2)
    }
    return data

if __name__ == "__main__":
    print(simulate_sensor())


# In[ ]:




