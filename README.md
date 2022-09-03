# SmartAd AB Testing User Analysis

10 Academy Batch 6 - Weekly Challenge: Week 2 - A/B Hypothesis Testing: Ad campaign performance

**Table of content**

- [Introduction](#introduction)
- [Overview](#overview)
- [Install](#install)
- [Data](#data)
- [Notebooks](#notebooks)
- [Models](#models)
- [Scripts](#scripts)
- [Test](#test)
- [Authors](#authors)

## Introduction

As a Machine learning engineer, our tasks is to design a reliable hypothesis testing algorithm for the BIO service and to determine whether a recent advertising campaign resulted in a significant lift in brand awareness.

## Overview

- Learning Outcomes
  > Statistical Modelling

> Using core data science python libraries pandas, matplotlib, seaborn, scikit-learn

> ML algorithms Logistic regression, Decision Trees, XGBoost

> Model management (building ML catalog contains model feature labels and training model version)

> MLOps with DVC, CML, and MLFlow

## Install

```
git clone https://github.com/Ad-Campaign-Performance/SmartAd_A-B_Testing_user_analysis.git
cd SmartAd_A-B_Testing_user_analysis
pip install -r requirements.txt
```

## Data

- The BIO data for this project is a “Yes” and “No” response of online users to the following question

> Do you know the brand Lux?

- Yes
- No

Data can be found [here at google drive](https://drive.google.com/file/d/1gJWvtl6roO7XMGLSfkMOFR3-D_yZvV63/view?usp=sharing)

The data collected for this challenge has the following columns

- auction_id: the unique id of the online user who has been presented the BIO.
- experiment: which group the user belongs to - control or exposed.
- date: the date in YYYY-MM-DD format
- hour: the hour of the day in HH format.
- device_make: the name of the type of device the user has e.g. Samsung
- platform_os: the id of the OS the user has.
- browser: the name of the browser the user uses to see the BIO questionnaire.
- yes: 1 if the user chooses the “Yes” radio button for the BIO questionnaire.
- no: 1 if the user chooses the “No” radio button for the BIO questionnaire.

## Notebooks

> All the Data Processing, EDA, statistical and sequential Analysis notebook file can be found in the notebooks folder.

## Models

> All the models generated will be found here in the models folder.
> All the databases schema will be found here in the models folder.

## Scripts

> All the Utils for Data munipulation and Ploting can be found here

## Tests

> All the unit and integration tests are found here in the tests folder.

## Authors

👤 **Anteneh Tilaye**

- GitHub: [Anteneh Tilaye](https://github.com/AntenehTilaye)
- LinkedIn: [Anteneh Tilaye](https://www.linkedin.com/in/anteneh-tilaye-bb6770149/)
- Website: [Anteneh Tilaye Demo Porfolio](https://antenehtilaye.github.io/)

👤 **Birhanu Gebisa**

- GitHub: [Birhanu Gebisa](https://github.com/BirhanuGebisa)
- LinkedIn: [Birhanu Gebisa](https://www.linkedin.com/in/birhanu-gebisa2721/)
- Website: [Birhanu Gebisa Porfolio](https://github.com/BirhanuGebisa.github.io/)

👤 **Genet Shanko**

- GitHub: [Genet Shanko](https://github.com/gshanko125298))
- LinkedIn: [Genet Shanko](https://www.linkedin.com/in/genet-dekebo-24b34658/)
- Website: [Genet Shanko Demo Porfolio](https://github.com/Ad-Campaign-Performance/SmartAd_A-B_Testing_user_analysis/pull/11)

👤 **Yishak Tadele**

- GitHub: [Yishak Tadele](https://github.com/isaaclucky)
- LinkedIn: [Yishak Tadele](https://www.linkedin.com/in/yishak-tadele/)
- Website: [Yishak Tadele Porfolio](http://yishaktadele.freecluster.eu/)

## Show US your support

Give US a ⭐ if you like this project!
