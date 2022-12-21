# Project description

**What is the main question your project aims to answer?**

Can we predict the number of daily bike riders counted at a "bike counting station" in Freiburg, based on the local temperature information and information about whether the given day is a business day or weekend?

**Which dataset will you use?**

We derive daily bike rider counts at a counting station in Freiburg in 2021 from the hourly measurements given in: https://www.mobidata-bw.de/dataset/eco-counter-fahrradzahler.

We combine this data with temperature data taken from https://open-meteo.com/en/docs/historical-weather-api.

(Business-day vs. weekend: Whether a given date is a business day or weekend can be inferred computationally from the date.)

**What analysis will you perform?**

We construct a linear regression model, assuming the number of counted bikeriders per day is a normally distributed variable depending on the mean temperature on the given day and whether the given day is a business day. We fit the model using Bayesian inference.


# Contents
* **analyses**: contains scripts to build the dataset and to analyze the data
* **data**: contains the raw data as well as the combined dataset (bike rider counts, temperature and weekday information)