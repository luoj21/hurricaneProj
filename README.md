# Predicting Hurricane Wind Speeds

Hurricane wind speed forecasting is essential for helping people understand and prepare for the upcoming severe weather conditions. Since forecasting deals primarily with time series data, autoregressive and moving average models are a popular choice to predict wind speeds based off of lagged features. 

Here, various machine learning models are trained and tested for accomplishing the same purpose. More specifically, these models attempt to predict the tangential wind speed of a hurricane at a time stamp based off of various lagged features that can be *N* time stamps ago. Each model shows varying levels of performance depending on if lagged features are used or how much training data is used.

To get started, clone this repsitory with

```
git clone https://github.com/luoj21/hurricaneProj.git

```

and install dependencies:

```
pip install -r requirements.txt

```

