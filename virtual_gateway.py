import random

def get_sensor_data():
    wind_speed = round(random.uniform(3, 15), 2)
    theoretical_power = round(wind_speed * 200, 2)
    wind_direction = round(random.uniform(0, 360), 2)
    return {
        "Wind Speed": wind_speed,
        "Theoretical Power": theoretical_power,
        "Wind Direction": wind_direction
    }


