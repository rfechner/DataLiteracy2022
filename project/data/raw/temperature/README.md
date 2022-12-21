This folder contains the temperature data taken from https://open-meteo.com/en/docs/historical-weather-api.

Currently, it only contains temperature data for one location.  
The file name includes information on the position (latitude, longitude).

**NOTE**:  
I just realized that the position does not correspond to the counter site position. Difference up 0.25° between the two.  
Coordinates are rounded by https://open-meteo.com/en/docs/historical-weather-api.  
Quick fix: At least, mention this inaccuracy in our documentation.  
Better fix: Find another service for historical weather data.  
Maybe https://meteostat.net/de/ ? Conveniently, a Python library is available: https://dev.meteostat.net/python/#installation. (So far, I didn't test whether this is better than https://open-meteo.com/en/docs/historical-weather-api).