# World Happiness Report 2023 - Exploratory Data Analysis

The World Happiness Report is an annual publication that ranks countries based on their happiness levels, as measured by a range of economic, social, and political indicators. The report is produced by the United Nations Sustainable Development Solutions Network. It is designed to provide policymakers, academics, and the general public with insights into the factors that contribute to happiness and well-being around the world.

In this project, I'll be using World Happiness Report 2023 data to visually analyze what factors make people in a country happy. The dataset can be found [here](https://www.kaggle.com/datasets/atom1991/world-happiness-report-2023). 

## Setting up
We will start by importing the necessary libraries and setting style for our visualizations.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.facecolor'] = '#00000000'
```

## The World Happiness Report Dataset
Let's have a look at the dataset's head and tail to get an idea of what it's like.


```python
df = pd.read_csv('data/WHR2023.csv')
print(f"Shape of the data: {df.shape}")

display(df.head())
df.tail()
```

    Shape of the data: (137, 21)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>iso alpha</th>
      <th>Regional indicator</th>
      <th>Happiness score</th>
      <th>Standard error of ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>...</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>South Asia</td>
      <td>1.859</td>
      <td>0.033</td>
      <td>1.923</td>
      <td>1.795</td>
      <td>7.324</td>
      <td>0.341</td>
      <td>54.712</td>
      <td>...</td>
      <td>-0.081</td>
      <td>0.847</td>
      <td>1.778</td>
      <td>0.645</td>
      <td>0.000</td>
      <td>0.087</td>
      <td>0.000</td>
      <td>0.093</td>
      <td>0.059</td>
      <td>0.976</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>ALB</td>
      <td>Central and Eastern Europe</td>
      <td>5.277</td>
      <td>0.066</td>
      <td>5.406</td>
      <td>5.148</td>
      <td>9.567</td>
      <td>0.718</td>
      <td>69.150</td>
      <td>...</td>
      <td>-0.007</td>
      <td>0.878</td>
      <td>1.778</td>
      <td>1.449</td>
      <td>0.951</td>
      <td>0.480</td>
      <td>0.549</td>
      <td>0.133</td>
      <td>0.037</td>
      <td>1.678</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>DZA</td>
      <td>Middle East and North Africa</td>
      <td>5.329</td>
      <td>0.062</td>
      <td>5.451</td>
      <td>5.207</td>
      <td>9.300</td>
      <td>0.855</td>
      <td>66.549</td>
      <td>...</td>
      <td>-0.117</td>
      <td>0.717</td>
      <td>1.778</td>
      <td>1.353</td>
      <td>1.298</td>
      <td>0.409</td>
      <td>0.252</td>
      <td>0.073</td>
      <td>0.152</td>
      <td>1.791</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Argentina</td>
      <td>ARG</td>
      <td>Latin America and Caribbean</td>
      <td>6.024</td>
      <td>0.063</td>
      <td>6.147</td>
      <td>5.900</td>
      <td>9.959</td>
      <td>0.891</td>
      <td>67.200</td>
      <td>...</td>
      <td>-0.089</td>
      <td>0.814</td>
      <td>1.778</td>
      <td>1.590</td>
      <td>1.388</td>
      <td>0.427</td>
      <td>0.587</td>
      <td>0.088</td>
      <td>0.082</td>
      <td>1.861</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Armenia</td>
      <td>ARM</td>
      <td>Commonwealth of Independent States</td>
      <td>5.342</td>
      <td>0.066</td>
      <td>5.470</td>
      <td>5.213</td>
      <td>9.615</td>
      <td>0.790</td>
      <td>67.789</td>
      <td>...</td>
      <td>-0.155</td>
      <td>0.705</td>
      <td>1.778</td>
      <td>1.466</td>
      <td>1.134</td>
      <td>0.443</td>
      <td>0.551</td>
      <td>0.053</td>
      <td>0.160</td>
      <td>1.534</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>iso alpha</th>
      <th>Regional indicator</th>
      <th>Happiness score</th>
      <th>Standard error of ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>...</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132</th>
      <td>Uzbekistan</td>
      <td>UZB</td>
      <td>Commonwealth of Independent States</td>
      <td>6.014</td>
      <td>0.059</td>
      <td>6.130</td>
      <td>5.899</td>
      <td>8.948</td>
      <td>0.875</td>
      <td>65.301</td>
      <td>...</td>
      <td>0.230</td>
      <td>0.638</td>
      <td>1.778</td>
      <td>1.227</td>
      <td>1.347</td>
      <td>0.375</td>
      <td>0.740</td>
      <td>0.260</td>
      <td>0.208</td>
      <td>1.856</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Venezuela</td>
      <td>VEN</td>
      <td>Latin America and Caribbean</td>
      <td>5.211</td>
      <td>0.064</td>
      <td>5.336</td>
      <td>5.085</td>
      <td>5.527</td>
      <td>0.839</td>
      <td>64.050</td>
      <td>...</td>
      <td>0.128</td>
      <td>0.811</td>
      <td>1.778</td>
      <td>0.000</td>
      <td>1.257</td>
      <td>0.341</td>
      <td>0.369</td>
      <td>0.205</td>
      <td>0.084</td>
      <td>2.955</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Vietnam</td>
      <td>VNM</td>
      <td>Southeast Asia</td>
      <td>5.763</td>
      <td>0.052</td>
      <td>5.865</td>
      <td>5.662</td>
      <td>9.287</td>
      <td>0.821</td>
      <td>65.502</td>
      <td>...</td>
      <td>-0.004</td>
      <td>0.759</td>
      <td>1.778</td>
      <td>1.349</td>
      <td>1.212</td>
      <td>0.381</td>
      <td>0.741</td>
      <td>0.134</td>
      <td>0.122</td>
      <td>1.824</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Zambia</td>
      <td>ZMB</td>
      <td>Sub-Saharan Africa</td>
      <td>3.982</td>
      <td>0.094</td>
      <td>4.167</td>
      <td>3.797</td>
      <td>8.074</td>
      <td>0.694</td>
      <td>55.032</td>
      <td>...</td>
      <td>0.098</td>
      <td>0.818</td>
      <td>1.778</td>
      <td>0.914</td>
      <td>0.890</td>
      <td>0.095</td>
      <td>0.545</td>
      <td>0.189</td>
      <td>0.080</td>
      <td>1.270</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Zimbabwe</td>
      <td>ZWE</td>
      <td>Sub-Saharan Africa</td>
      <td>3.204</td>
      <td>0.061</td>
      <td>3.323</td>
      <td>3.084</td>
      <td>7.641</td>
      <td>0.690</td>
      <td>54.050</td>
      <td>...</td>
      <td>-0.046</td>
      <td>0.766</td>
      <td>1.778</td>
      <td>0.758</td>
      <td>0.881</td>
      <td>0.069</td>
      <td>0.363</td>
      <td>0.112</td>
      <td>0.117</td>
      <td>0.905</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



137 countries are included in the survey. Looks like the dataset have a few redundant fields. Let's explore further.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 137 entries, 0 to 136
    Data columns (total 21 columns):
     #   Column                                      Non-Null Count  Dtype  
    ---  ------                                      --------------  -----  
     0   Country name                                137 non-null    object 
     1   iso alpha                                   137 non-null    object 
     2   Regional indicator                          137 non-null    object 
     3   Happiness score                             137 non-null    float64
     4   Standard error of ladder score              137 non-null    float64
     5   upperwhisker                                137 non-null    float64
     6   lowerwhisker                                137 non-null    float64
     7   Logged GDP per capita                       137 non-null    float64
     8   Social support                              137 non-null    float64
     9   Healthy life expectancy                     136 non-null    float64
     10  Freedom to make life choices                137 non-null    float64
     11  Generosity                                  137 non-null    float64
     12  Perceptions of corruption                   137 non-null    float64
     13  Ladder score in Dystopia                    137 non-null    float64
     14  Explained by: Log GDP per capita            137 non-null    float64
     15  Explained by: Social support                137 non-null    float64
     16  Explained by: Healthy life expectancy       136 non-null    float64
     17  Explained by: Freedom to make life choices  137 non-null    float64
     18  Explained by: Generosity                    137 non-null    float64
     19  Explained by: Perceptions of corruption     137 non-null    float64
     20  Dystopia + residual                         136 non-null    float64
    dtypes: float64(18), object(3)
    memory usage: 22.6+ KB
    

Now we will filter the columns that will be used for our analysis. We are dropping "Explained by" columns as they are processed scores, as well as "Ladder score in Dystopia" since it is a constant value. We will also drop Happiness score's statistics as they are not needed for our visualizations.


```python
cols = ['Country name', 'iso alpha', 'Regional indicator', 'Happiness score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
df = df.filter(cols)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>iso alpha</th>
      <th>Regional indicator</th>
      <th>Happiness score</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>South Asia</td>
      <td>1.859</td>
      <td>7.324</td>
      <td>0.341</td>
      <td>54.712</td>
      <td>0.382</td>
      <td>-0.081</td>
      <td>0.847</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>ALB</td>
      <td>Central and Eastern Europe</td>
      <td>5.277</td>
      <td>9.567</td>
      <td>0.718</td>
      <td>69.150</td>
      <td>0.794</td>
      <td>-0.007</td>
      <td>0.878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>DZA</td>
      <td>Middle East and North Africa</td>
      <td>5.329</td>
      <td>9.300</td>
      <td>0.855</td>
      <td>66.549</td>
      <td>0.571</td>
      <td>-0.117</td>
      <td>0.717</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Argentina</td>
      <td>ARG</td>
      <td>Latin America and Caribbean</td>
      <td>6.024</td>
      <td>9.959</td>
      <td>0.891</td>
      <td>67.200</td>
      <td>0.823</td>
      <td>-0.089</td>
      <td>0.814</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Armenia</td>
      <td>ARM</td>
      <td>Commonwealth of Independent States</td>
      <td>5.342</td>
      <td>9.615</td>
      <td>0.790</td>
      <td>67.789</td>
      <td>0.796</td>
      <td>-0.155</td>
      <td>0.705</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Country name'].unique()
```




    array(['Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Armenia',
           'Australia', 'Austria', 'Bahrain', 'Bangladesh', 'Belgium',
           'Benin', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil',
           'Bulgaria', 'Burkina Faso', 'Cambodia', 'Cameroon', 'Canada',
           'Chad', 'Chile', 'China', 'Colombia', 'Comoros',
           'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Costa Rica', 'Croatia',
           'Cyprus', 'Czechia', 'Denmark', 'Dominican Republic', 'Ecuador',
           'Egypt', 'El Salvador', 'Estonia', 'Ethiopia', 'Finland', 'France',
           'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece',
           'Guatemala', 'Guinea', 'Honduras', 'Hong Kong S.A.R. of China',
           'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq',
           'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan',
           'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kyrgyzstan', 'Laos',
           'Latvia', 'Lebanon', 'Liberia', 'Lithuania', 'Luxembourg',
           'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Malta', 'Mauritania',
           'Mauritius', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro',
           'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal',
           'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
           'North Macedonia', 'Norway', 'Pakistan', 'Panama', 'Paraguay',
           'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia',
           'Saudi Arabia', 'Senegal', 'Serbia', 'Sierra Leone', 'Singapore',
           'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain',
           'Sri Lanka', 'State of Palestine', 'Sweden', 'Switzerland',
           'Taiwan Province of China', 'Tajikistan', 'Tanzania', 'Thailand',
           'Togo', 'Tunisia', 'Turkiye', 'Uganda', 'Ukraine',
           'United Arab Emirates', 'United Kingdom', 'United States',
           'Uruguay', 'Uzbekistan', 'Venezuela', 'Vietnam', 'Zambia',
           'Zimbabwe'], dtype=object)



As seen above, a few countries including Palestine, Hong Kong, and Taiwan have longer names. Let's shorten them for our ease later.


```python
rename_dict = {
    'Taiwan Province of China': 'Taiwan',
    'Hong Kong S.A.R. of China': 'Hong Kong',
    'State of Palestine': 'Palestine'
}
df['Country name'].replace(rename_dict, inplace=True)
```

Let's have a look at the statistics of our dataset using a custom function that combines `describe` and `info` methods.


```python
def describe(df):    
    # Show information better than describe() and info()
    desc = pd.DataFrame(index=df.columns)
    desc["count"] = df.count()
    desc["null"] = df.isna().sum()
    desc["%null"] = desc["null"] / len(df) * 100
    desc["nunique"] = df.nunique()
    desc["%unique"] = desc["nunique"] / len(df) * 100
    desc["type"] = df.dtypes
    desc = pd.concat([desc, df.describe().T.drop("count", axis=1)], axis=1)

    return desc

describe(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>null</th>
      <th>%null</th>
      <th>nunique</th>
      <th>%unique</th>
      <th>type</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Country name</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>137</td>
      <td>100.000000</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>iso alpha</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>137</td>
      <td>100.000000</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Regional indicator</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>10</td>
      <td>7.299270</td>
      <td>object</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Happiness score</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>134</td>
      <td>97.810219</td>
      <td>float64</td>
      <td>5.539796</td>
      <td>1.139929</td>
      <td>1.859</td>
      <td>4.7240</td>
      <td>5.6840</td>
      <td>6.3340</td>
      <td>7.804</td>
    </tr>
    <tr>
      <th>Logged GDP per capita</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>135</td>
      <td>98.540146</td>
      <td>float64</td>
      <td>9.449796</td>
      <td>1.207302</td>
      <td>5.527</td>
      <td>8.5910</td>
      <td>9.5670</td>
      <td>10.5400</td>
      <td>11.660</td>
    </tr>
    <tr>
      <th>Social support</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>116</td>
      <td>84.671533</td>
      <td>float64</td>
      <td>0.799073</td>
      <td>0.129222</td>
      <td>0.341</td>
      <td>0.7220</td>
      <td>0.8270</td>
      <td>0.8960</td>
      <td>0.983</td>
    </tr>
    <tr>
      <th>Healthy life expectancy</th>
      <td>136</td>
      <td>1</td>
      <td>0.729927</td>
      <td>125</td>
      <td>91.240876</td>
      <td>float64</td>
      <td>64.967632</td>
      <td>5.750390</td>
      <td>51.530</td>
      <td>60.6485</td>
      <td>65.8375</td>
      <td>69.4125</td>
      <td>77.280</td>
    </tr>
    <tr>
      <th>Freedom to make life choices</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>117</td>
      <td>85.401460</td>
      <td>float64</td>
      <td>0.787394</td>
      <td>0.112371</td>
      <td>0.382</td>
      <td>0.7240</td>
      <td>0.8010</td>
      <td>0.8740</td>
      <td>0.961</td>
    </tr>
    <tr>
      <th>Generosity</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>122</td>
      <td>89.051095</td>
      <td>float64</td>
      <td>0.022431</td>
      <td>0.141707</td>
      <td>-0.254</td>
      <td>-0.0740</td>
      <td>0.0010</td>
      <td>0.1170</td>
      <td>0.531</td>
    </tr>
    <tr>
      <th>Perceptions of corruption</th>
      <td>137</td>
      <td>0</td>
      <td>0.000000</td>
      <td>115</td>
      <td>83.941606</td>
      <td>float64</td>
      <td>0.725401</td>
      <td>0.176956</td>
      <td>0.146</td>
      <td>0.6680</td>
      <td>0.7740</td>
      <td>0.8460</td>
      <td>0.929</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.isna().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>iso alpha</th>
      <th>Regional indicator</th>
      <th>Happiness score</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>116</th>
      <td>Palestine</td>
      <td>PSE</td>
      <td>Middle East and North Africa</td>
      <td>4.908</td>
      <td>8.716</td>
      <td>0.859</td>
      <td>NaN</td>
      <td>0.694</td>
      <td>-0.132</td>
      <td>0.836</td>
    </tr>
  </tbody>
</table>
</div>



A missing value has been identified in the Healthy life expectancy column for Palestine. I noted this for future reference but decided to leave the value as is, as imputation could bias the information. Leaving this missing value will not affect the analysis.

## Visualizations

### Relationship between factors

Using `sns.pairplot()` to have a look on relationship among variables. There seems to be a strong positive relationship among most of them.


```python
sns.set_theme(style="ticks")
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x1a07fc30090>




    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_20_1.png)
    


Let's plot a heatmap of correlation among variables to have a clearer look on their relationship


```python
plt.figure(figsize=(11, 9))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
```


    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_22_0.png)
    


- The scatter plots and heat map reveals several strong correlations between factors and happiness score. It has been found that countries with higher GDP per capita, healthier life expectancies, greater freedom to make choices, and stronger social support tend to have higher happiness scores. Furthermore, a strong positive correlation has been found between GDP per capita, healthy life expectancy, and social support.
- In addition, a negative correlation has been found between happiness score and perceptions of corruption. This suggests that maintaining high levels of happiness among citizens may be challenging for countries with higher levels of corruption.
- Overall, these findings highlight the importance of economic, social, and political factors in determining happiness levels across countries.

**Top ten happiest countries in the world**


```python
happiest = df[['Country name', 'Happiness score']].sort_values('Happiness score', ascending=False).head(10)

plt.figure(figsize=(15, 5))
plt.title('Top 10 happiest countries in the World')
sns.barplot(happiest, x='Country name', y='Happiness score', palette='muted')
plt.xlabel(None)
```




    Text(0.5, 0, '')




    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_25_1.png)
    


Finland, Denmark, Iceland, Israel, Netherlands, Sweden, Norway, Switzerland, Luxembourg, and New Zealand are clearly the top happiest countries in the World. Follow along to explore what makes them the happiest.

**Which countries are best in each factors?**


```python
factors = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Top 10 countries with highest:', fontsize=16)

for i, ax in enumerate(fig.axes):
    factor = factors[i]
    ax.set_title(factor)
    ax.tick_params(labelrotation=75)

    negative = (factor == 'Perceptions of corruption')
    highest_factor = df[['Country name', factor]].sort_values(factor, ascending=negative).head(10)
    sns.barplot(highest_factor, x='Country name', y=factor, ax=ax, palette='muted')
    
    ax.set_xlabel(None)

plt.tight_layout(pad=2);
```


    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_28_0.png)
    


Top 10 countries in all the factors vary. However, the thing to notice is that there are countries among all the factors that were absent in happiest countries plot. Maybe try some other way to identify what makes people in a country happy. But before that, let's have a look on least happy countries.


```python
least_happy = df[['Country name', 'Happiness score']].sort_values('Happiness score', ascending=True).head(10)

plt.figure(figsize=(15, 5))
plt.title('Top 10 least happy countries in the World')
sns.barplot(least_happy, x='Country name', y='Happiness score', palette='muted');
plt.xlabel(None)
```




    Text(0.5, 0, '')




    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_30_1.png)
    


Afghanistan, Lebanon, Sierra Leone, Zimbabwe, Congo (Kinshasa), Botswana, Malawi, Comoros, Tanzania, and Zambia are the least happy countries.


```python
factors = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Top 10 countries with highest:', fontsize=16)

for i, ax in enumerate(fig.axes):
    factor = factors[i]
    ax.set_title(factor)
    ax.tick_params(labelrotation=75)

    negative = (factor == 'Perceptions of corruption')
    highest_factor = df[['Country name', factor]].sort_values(factor, ascending=True ^ negative).head(10)
    sns.barplot(highest_factor, x='Country name', y=factor, ax=ax, palette='muted')
    
    ax.set_xlabel(None)

plt.tight_layout(pad=2)
```


    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_32_0.png)
    


Afghanistan being the least happy country has also least freedom to make life choices and social support but one of the highest GDP per capita among the least happy countries. Again, we can see some other countries in all these plots therefore, have a look at scatter plots of all the factors against happiness score. 

**Let's plot the relationship between happiness score and all the factors in one figure for ease of us to compare**


```python
factors = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
tab_20_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#bcbd22", "#9edae5"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Relationship between happiness score and other factors', fontsize=16)

for i, ax in enumerate(fig.axes):
    factor = factors[i]

    corr = df['Happiness score'].corr(df[factor])
    ax.set_title(f'Correlation = {corr:.4f}')

    sns.regplot(df, x='Happiness score', y=factor, ax=ax, color=tab_20_colors[i])

plt.tight_layout(pad=2)
```


    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_35_0.png)
    


- As expected, GDP per capita, Social support and Healthy life expectancy are strong contributors to happiness.
- Freedom to make life choices also has a positive impact but less than the other factors.
- Correlation between happiness score and generosity is only 4.4%. One thing that can be inferred from it is that maybe, the happiest countries have the least proportion of population that's needy and hence they are not necessarily the most generous nations.
- It's clearly visible that countries with highest corruption are least happy. However, the relationship doesn't seem too linear as it can be seen above. Correlation between happiness score and perceptions of corruption is only -47%.

### Macro geo-spatial analysis

We can start by plotting the happiness score of countries on a world map to get a rough overview. We will use the `plotly` library to create an interactive map.


```python
fig = px.choropleth(
    df, 
    locations='iso alpha', 
    color='Happiness score',
    hover_name='Country name',
    title='Global Happiness Map',
    color_continuous_scale=px.colors.diverging.RdBu)

fig.update_layout(
    margin=dict(l=50, r=0, b=0, t=100),
    width=1300, 
    height=700)
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v2.29.1
* Copyright 2012-2024, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
/*! For license information please see plotly.min.js.LICENSE.txt */
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>




<div>                            <div id="89e88e88-b742-437e-8c91-531187cee337" class="plotly-graph-div" style="height:700px; width:1300px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("89e88e88-b742-437e-8c91-531187cee337")) {                    Plotly.newPlot(                        "89e88e88-b742-437e-8c91-531187cee337",                        [{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eiso alpha=%{location}\u003cbr\u003eHappiness score=%{z}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Afghanistan","Albania","Algeria","Argentina","Armenia","Australia","Austria","Bahrain","Bangladesh","Belgium","Benin","Bolivia","Bosnia and Herzegovina","Botswana","Brazil","Bulgaria","Burkina Faso","Cambodia","Cameroon","Canada","Chad","Chile","China","Colombia","Comoros","Congo (Brazzaville)","Congo (Kinshasa)","Costa Rica","Croatia","Cyprus","Czechia","Denmark","Dominican Republic","Ecuador","Egypt","El Salvador","Estonia","Ethiopia","Finland","France","Gabon","Gambia","Georgia","Germany","Ghana","Greece","Guatemala","Guinea","Honduras","Hong Kong","Hungary","Iceland","India","Indonesia","Iran","Iraq","Ireland","Israel","Italy","Ivory Coast","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kosovo","Kyrgyzstan","Laos","Latvia","Lebanon","Liberia","Lithuania","Luxembourg","Madagascar","Malawi","Malaysia","Mali","Malta","Mauritania","Mauritius","Mexico","Moldova","Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nepal","Netherlands","New Zealand","Nicaragua","Niger","Nigeria","North Macedonia","Norway","Pakistan","Panama","Paraguay","Peru","Philippines","Poland","Portugal","Romania","Russia","Saudi Arabia","Senegal","Serbia","Sierra Leone","Singapore","Slovakia","Slovenia","South Africa","South Korea","Spain","Sri Lanka","Palestine","Sweden","Switzerland","Taiwan","Tajikistan","Tanzania","Thailand","Togo","Tunisia","Turkiye","Uganda","Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay","Uzbekistan","Venezuela","Vietnam","Zambia","Zimbabwe"],"locations":["AFG","ALB","DZA","ARG","ARM","AUS","AUT","BHR","BGD","BEL","BEN","BOL","BIH","BWA","BRA","BGR","BFA","KHM","CMR","CAN","TCD","CHL","CHN","COL","COM","COG","COD","CRI","HRV","CYP","CZE","DNK","DOM","ECU","EGY","SLV","EST","ETH","FIN","FRA","GAB","GMB","GEO","DEU","GHA","GRC","GTM","GIN","HND","HKG","HUN","ISL","IND","IDN","IRN","IRQ","IRL","ISR","ITA","CIV","JAM","JPN","JOR","KAZ","KEN","XK","KGZ","LAO","LVA","LBN","LBR","LTU","LUX","MDG","MWI","MYS","MLI","MLT","MRT","MUS","MEX","MDA","MNG","MNE","MAR","MOZ","MMR","NAM","NPL","NLD","NZL","NIC","NER","NGA","MKD","NOR","PAK","PAN","PRY","PER","PHL","POL","PRT","ROU","RUS","SAU","SEN","SRB","SLE","SGP","SVK","SVN","ZAF","KOR","ESP","LKA","PSE","SWE","CHE","TWN","TJK","TZA","THA","TGO","TUN","TUR","UGA","UKR","ARE","GBR","USA","URY","UZB","VEN","VNM","ZMB","ZWE"],"name":"","z":[1.859,5.277,5.329,6.024,5.342,7.095,7.097,6.173,4.282,6.859,4.374,5.684,5.633,3.435,6.125,5.466,4.638,4.393,4.973,6.961,4.397,6.334,5.818,5.63,3.545,5.267,3.207,6.609,6.125,6.13,6.845,7.586,5.569,5.559,4.17,6.122,6.455,4.091,7.804,6.661,5.035,4.279,5.109,6.892,4.605,5.931,6.15,5.072,6.023,5.308,6.041,7.53,4.036,5.277,4.876,4.941,6.911,7.473,6.405,5.053,5.703,6.129,4.12,6.144,4.487,6.368,5.825,5.111,6.213,2.392,4.042,6.763,7.228,4.019,3.495,6.012,4.198,6.3,4.724,5.902,6.33,5.819,5.84,5.722,4.903,4.954,4.372,4.631,5.36,7.403,7.123,6.259,4.501,4.981,5.254,7.315,4.555,6.265,5.738,5.526,5.523,6.26,5.968,6.589,5.661,6.463,4.855,6.144,3.138,6.587,6.469,6.65,5.275,5.951,6.436,4.442,4.908,7.395,7.24,6.535,5.33,3.694,5.843,4.137,4.497,4.614,4.432,5.071,6.571,6.796,6.894,6.494,6.014,5.211,5.763,3.982,3.204],"type":"choropleth"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{}},"coloraxis":{"colorbar":{"title":{"text":"Happiness score"}},"colorscale":[[0.0,"rgb(103,0,31)"],[0.1,"rgb(178,24,43)"],[0.2,"rgb(214,96,77)"],[0.3,"rgb(244,165,130)"],[0.4,"rgb(253,219,199)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(209,229,240)"],[0.7,"rgb(146,197,222)"],[0.8,"rgb(67,147,195)"],[0.9,"rgb(33,102,172)"],[1.0,"rgb(5,48,97)"]]},"legend":{"tracegroupgap":0},"title":{"text":"Global Happiness Map"},"margin":{"l":50,"r":0,"b":0,"t":100},"width":1300,"height":700},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('89e88e88-b742-437e-8c91-531187cee337');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


There is a clear happiness bias between regions, let's see what exactly is the case.


```python
plt.figure(figsize=(10, 7))
plt.title("Happiness Score by region", size=16)
sns.boxplot(df, x='Happiness score', y='Regional indicator', palette='tab20')
plt.ylabel(None)
```




    Text(0, 0.5, '')




    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_41_1.png)
    


- The box plot analysis indicates that higher median happiness scores are observed in the Western Europe, North America, and ANZ regions compared to other regions. These are developed regions with more established economies and social systems, which may explain the trend.
- An outlier is observed in the South Asia region, where Afghanistan has a notably lower happiness score. This may be attributed to political instability and hence lower social support and freedom in Afghanistan even though it's GDP per capita is not the worst, as we discussed.

## Conclusion

The comprehensive analysis of happiness scores around the world highlights the importance of economic, social, and health factors in determining societal well-being. The findings suggests that GDP per capita, Social support, Healthy life expectancy, and Freedom to make life choices has the highest impact on happiness levels, while reducing corruption can help improve stability. Hence, top happiest countries can be seen among the top ten countries in all these factors. Generosity seem to be a less important factor as the data speaks for itself. 

By prioritizing happiness as a key goal for individuals, communities, and policymakers, we can work towards creating a world that is more just, equitable, and fulfilling for all.

## Extra: Machine Learning for EDA
We can use baseline ML models to anticipate feature importances. This is a really nice feature of supervised learning beyond predictive analysis.

One of the best models for extracting feature importances is Random Forest as it is very robust yet simple.

**Preparing and training a simple Random Forest**


```python
from sklearn.ensemble import RandomForestRegressor
SEED = 42
```


```python
# Drop the categorical columns to make it simple
X = df.drop(columns=['Country name', 'iso alpha', 'Regional indicator', 'Happiness score'])
y = df['Happiness score']

# We need to fill in the missing value found earlier here in order for the model to work
X['Healthy life expectancy'].fillna(X['Healthy life expectancy'].median(), inplace=True)
```


```python
rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
rf.fit(X, y)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(random_state=42)</pre></div></div></div></div></div>



**The Feature Importances plot**


```python
plt.figure(figsize=(10, 6))
plt.title('Random Forest Feature Importances')
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.barh(feat_imp.index, feat_imp.values)
```




    <BarContainer object of 6 artists>




    
![png](EDA-world-happiness-report-2023_files/EDA-world-happiness-report-2023_50_1.png)
    


Quick and simple feature importance evaluation confirms our finding earlier, amazing! More could have been done but that would be beyond the scope of this project.