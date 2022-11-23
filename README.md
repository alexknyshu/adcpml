---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="kl6qF2_nHALT"}
# A machine learning algorithm for near-shore ADCP data processing

The following machine learning study aims to select several typical and
several anomalous loadcases (about 20-30 loadcases in total) from a
detailed ADCP datasets obtained at the Wood Island research site, Maine,
USA. Two ADCPs measured water depth, significant wave height and period,
as well as north and east projections of current velocity profiles (15
minute averages) during a half of a month period. More details on the
research can be found in [Section 3.2 of this
paper](https://github.com/alexanderknysh/adcpml/blob/main/paper.pdf).

## Data formatting

Before we start processing the ADCP datasets, let\'s first list the
libraries needed for the analysis
:::

::: {.cell .code id="eWZetvLUZXcy"}
``` {.python}
# required libraries
import pandas              as pd
import numpy               as np
import math                as m
import matplotlib.pyplot   as plt
import plotly.express      as px
from collections           import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster       import KMeans
from sklearn               import metrics
```
:::

::: {.cell .markdown id="g3L74K0GRfQY"}
and define a function that will format velocity profiles to the total
fluid energy in the upper ocean layers:
:::

::: {.cell .code id="owDavpEqQSCr"}
``` {.python}
# function that format velocity profile datasets
def format_profile(data, top): 
  output = np.empty((0,top))
  for i in range(data.shape[0]):
    row = data[i]
    row = row[np.logical_not(np.isnan(row))]
    row = row[-top:]
    row = np.dot(row, row)
    output = np.append(output, row)
  return output
```
:::

::: {.cell .markdown id="5gNNDPH-SVhF"}
Next, upload the datasets from the [following GitHub
repository](https://github.com/alexanderknysh/adcpml):
:::

::: {.cell .code id="jvNYJ0s0QZ9-"}
``` {.python}
# upload datasets common for both adcps:  wave properties and water depth
adcpdata = pd.read_excel('https://github.com/alexanderknysh/adcpml/blob/main/data_adcp_cm.xlsx?raw=true')

# upload velocity dataset: west and east adpc profiles
west_vx  = pd.read_excel('https://github.com/alexanderknysh/adcpml/blob/main/west_adcp_vx.xlsx?raw=true')
west_vy  = pd.read_excel('https://github.com/alexanderknysh/adcpml/blob/main/west_adcp_vy.xlsx?raw=true')
east_vx  = pd.read_excel('https://github.com/alexanderknysh/adcpml/blob/main/east_adcp_vx.xlsx?raw=true')
east_vy  = pd.read_excel('https://github.com/alexanderknysh/adcpml/blob/main/east_adcp_vy.xlsx?raw=true')

# other important data
samples = range(0, adcpdata.shape[0]) # range of field samples
alpha   = 13*m.pi/180                 # major axis of tidal ellipse (13 degrees)
cells   = 9                           # number of velocity measurement cells
```
:::

::: {.cell .markdown id="rIVZx04nW42z"}
In ocean engineering, tidal-driven current velocities are usually
represented in terms of projections on major and minor axes of a tidal
ellipse. In this study, we are mostly interested in the major
projections since they have the highest absolute values of current
velocities. Both west and east major velocity profiles are converted to
the energy values that represent total kinetic energy in the upper water
layers (4 meters deep). This also reduces number of features we have to
deal with in the future.
:::

::: {.cell .code colab="{\"height\":424,\"base_uri\":\"https://localhost:8080/\"}" id="mQDODmuAQzvU" outputId="3e9529f9-39b5-4854-f0fa-cdf1ecb4efce"}
``` {.python}
# project the velocity profiles on the major axis of the tidal ellipse
# save as numpy array to ease further formatting
# convert profiles to relative energy
west_major = west_vx * m.cos(alpha) - west_vy * m.sin(alpha)
east_major = east_vx * m.cos(alpha) - east_vy * m.sin(alpha)
west_major = west_major.to_numpy()
east_major = east_major.to_numpy()
adcpdata['WestEnergy'] = format_profile(west_major, cells)
adcpdata['EastEnergy'] = format_profile(east_major, cells)

# display the resulting dataset
display(adcpdata)
```

::: {.output .display_data}
```{=html}
  <div id="df-9ef24e67-b993-40ac-927c-708707095716">
    <div class="colab-df-container">
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
      <th>Date</th>
      <th>WaterDepth</th>
      <th>WaveHeight</th>
      <th>WavePeriod</th>
      <th>WestEnergy</th>
      <th>EastEnergy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-05-16 14:15:00.000</td>
      <td>9.475</td>
      <td>0.235</td>
      <td>11.075</td>
      <td>0.174540</td>
      <td>0.136834</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-05-16 14:30:00.000</td>
      <td>9.395</td>
      <td>0.250</td>
      <td>12.150</td>
      <td>0.271423</td>
      <td>0.177364</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-05-16 14:44:59.990</td>
      <td>9.289</td>
      <td>0.240</td>
      <td>11.370</td>
      <td>0.249177</td>
      <td>0.214879</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-05-16 14:59:59.985</td>
      <td>9.160</td>
      <td>0.235</td>
      <td>11.555</td>
      <td>0.355531</td>
      <td>0.250243</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-05-16 15:14:59.980</td>
      <td>9.007</td>
      <td>0.210</td>
      <td>11.640</td>
      <td>0.471080</td>
      <td>0.328769</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1157</th>
      <td>2019-05-28 15:29:54.215</td>
      <td>7.275</td>
      <td>0.275</td>
      <td>8.780</td>
      <td>0.025099</td>
      <td>0.019459</td>
    </tr>
    <tr>
      <th>1158</th>
      <td>2019-05-28 15:44:54.210</td>
      <td>7.176</td>
      <td>0.260</td>
      <td>8.640</td>
      <td>0.063899</td>
      <td>0.029413</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>2019-05-28 15:59:54.205</td>
      <td>7.083</td>
      <td>0.260</td>
      <td>8.660</td>
      <td>0.072142</td>
      <td>0.073600</td>
    </tr>
    <tr>
      <th>1160</th>
      <td>2019-05-28 16:14:54.200</td>
      <td>7.040</td>
      <td>0.280</td>
      <td>8.360</td>
      <td>0.097749</td>
      <td>0.094061</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>2019-05-28 16:29:54.195</td>
      <td>6.987</td>
      <td>0.275</td>
      <td>8.350</td>
      <td>0.143435</td>
      <td>0.082282</td>
    </tr>
  </tbody>
</table>
<p>1162 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9ef24e67-b993-40ac-927c-708707095716')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9ef24e67-b993-40ac-927c-708707095716 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9ef24e67-b993-40ac-927c-708707095716');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="0dqilF11eI8N"}
Let\'s now visualize the final dataset using Matplotlib:
:::

::: {.cell .code colab="{\"height\":759,\"base_uri\":\"https://localhost:8080/\"}" id="YWDE53DBea8M" outputId="b8fe594f-ffe9-4675-a345-a690133edd61"}
``` {.python}
# visualize the adcp data
fig, axs = plt.subplots(5, figsize=(17, 13))

axs[0].plot(samples, adcpdata.iloc[:,1], c='blue', label='Water depth, $(m)$')
axs[0].set_ylim([0, 15])
axs[0].legend(loc='upper right')

axs[1].plot(samples, adcpdata.iloc[:,2], c='purple', label='Significant wave height, $(m)$')
axs[1].set_ylim([0, 1])
axs[1].legend(loc='upper right')

axs[2].plot(samples, adcpdata.iloc[:,3], c='gray', label='Significant wave period, $(s)$')
axs[2].set_ylim([0, 20])
axs[2].legend(loc='upper right')

axs[3].plot(samples, adcpdata.iloc[:,4], c='green', label='Current energy at west ADCP, $(m/s)^2$')
axs[3].set_ylim([0, 2.5])
axs[3].legend(loc='upper right')

axs[4].plot(samples, adcpdata.iloc[:,5], c='red', label='Current energy at east ADCP, $(m/s)^2$')
axs[4].set_ylim([0, 2.5])
axs[4].legend(loc='upper right')

plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/a8ba21e6c236ea598b3cbb5a7c17ab2774a6840c.png)
:::
:::

::: {.cell .markdown id="BpWhPbT87bj-"}
## Data scaling

Even though we have processed the raw ADCP data, it must be still
modified before using a clustering algorithm. The problem that is ranges
of the feature values vary significantly, and it may cause machine
learning algorythms to perform poorly. Thus, the so-called **Min-Max
scaling** is applied to make all features vary from 0 to 1
(normalization) since there are no significant outliers in the dataset.
:::

::: {.cell .code colab="{\"height\":424,\"base_uri\":\"https://localhost:8080/\"}" id="NgI1rKpR7cEl" outputId="d5910c46-7e41-4d54-d37c-b6568ec49dcc"}
``` {.python}
# remove the date column and apply min-max scaling
adcpdata_nodate = adcpdata.drop('Date', axis='columns')
scaler = StandardScaler()
scaler.fit(adcpdata_nodate)
adcpdata_scaled = scaler.fit_transform(adcpdata_nodate)
adcpdata_scaled = pd.DataFrame(adcpdata_scaled, columns=adcpdata_nodate.columns)

# display scaled data
display(adcpdata_scaled)
```

::: {.output .display_data}
```{=html}
  <div id="df-cb08671f-fb98-4b15-8fe3-ff7d41f5affe">
    <div class="colab-df-container">
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
      <th>WaterDepth</th>
      <th>WaveHeight</th>
      <th>WavePeriod</th>
      <th>WestEnergy</th>
      <th>EastEnergy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.422928</td>
      <td>-0.449727</td>
      <td>1.936302</td>
      <td>-0.387866</td>
      <td>-0.515059</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.343870</td>
      <td>-0.326437</td>
      <td>2.536732</td>
      <td>-0.033365</td>
      <td>-0.373893</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.239117</td>
      <td>-0.408630</td>
      <td>2.101071</td>
      <td>-0.114764</td>
      <td>-0.243228</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.111635</td>
      <td>-0.449727</td>
      <td>2.204401</td>
      <td>0.274392</td>
      <td>-0.120053</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.960436</td>
      <td>-0.655209</td>
      <td>2.251877</td>
      <td>0.697193</td>
      <td>0.153459</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1157</th>
      <td>-0.751181</td>
      <td>-0.120955</td>
      <td>0.654453</td>
      <td>-0.934683</td>
      <td>-0.923883</td>
    </tr>
    <tr>
      <th>1158</th>
      <td>-0.849016</td>
      <td>-0.244245</td>
      <td>0.576258</td>
      <td>-0.792711</td>
      <td>-0.889214</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>-0.940921</td>
      <td>-0.244245</td>
      <td>0.587428</td>
      <td>-0.762548</td>
      <td>-0.735306</td>
    </tr>
    <tr>
      <th>1160</th>
      <td>-0.983415</td>
      <td>-0.079859</td>
      <td>0.419867</td>
      <td>-0.668851</td>
      <td>-0.664039</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>-1.035792</td>
      <td>-0.120955</td>
      <td>0.414281</td>
      <td>-0.501682</td>
      <td>-0.705066</td>
    </tr>
  </tbody>
</table>
<p>1162 rows × 5 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cb08671f-fb98-4b15-8fe3-ff7d41f5affe')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cb08671f-fb98-4b15-8fe3-ff7d41f5affe button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cb08671f-fb98-4b15-8fe3-ff7d41f5affe');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="09ed5GbIAGUv"}
Take a look at the scaled data.
:::

::: {.cell .code colab="{\"height\":759,\"base_uri\":\"https://localhost:8080/\"}" id="WrjgrPkB_Orw" outputId="c9053956-24eb-4079-9ca8-1f02adbc69b3"}
``` {.python}
# visualize the adcp data
fig, axs = plt.subplots(5, figsize=(17, 13))

axs[0].plot(samples, adcpdata_scaled.iloc[:,0], c='blue', label='Water depth')
axs[0].set_ylim([-2, 2])
axs[0].legend(loc='upper right')

axs[1].plot(samples, adcpdata_scaled.iloc[:,1], c='purple', label='Significant wave height')
axs[1].set_ylim([-2, 6])
axs[1].legend(loc='upper right')

axs[2].plot(samples, adcpdata_scaled.iloc[:,2], c='gray', label='Significant wave period')
axs[2].set_ylim([-4, 4])
axs[2].legend(loc='upper right')

axs[3].plot(samples, adcpdata_scaled.iloc[:,3], c='green', label='Current energy at west ADCP')
axs[3].set_ylim([-2, 8])
axs[3].legend(loc='upper right')

axs[4].plot(samples, adcpdata_scaled.iloc[:,4], c='red', label='Current energy at east ADCP')
axs[4].set_ylim([-2, 8])
axs[4].legend(loc='upper right')

plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/1cfa0f25c33f479045f6c4b9793f88bb7adcb81d.png)
:::
:::

::: {.cell .markdown id="7AjeBzUAANU0"}
While there is not that much correlation between wated depth,
significant wave height and significant wave period, the connection
between water depth and current energy is very clear. Let\'s zoom in to
see high and low tides:
:::

::: {.cell .code colab="{\"height\":324,\"base_uri\":\"https://localhost:8080/\"}" id="qzLEkjUCA6-7" outputId="2df62f72-bf07-408c-dbe6-9b6308276d77"}
``` {.python}
# water depth vs current energy
fig, axs = plt.subplots(figsize=(17, 5))

axs.plot(samples, adcpdata_scaled.iloc[:,0], c='blue', label='Water depth')
axs.plot(samples, adcpdata_scaled.iloc[:,3], c='green', label='Current energy at west ADCP')
axs.plot(samples, adcpdata_scaled.iloc[:,4], c='red', label='Current energy at east ADCP')
axs.set_xlim([0, 400])
axs.set_ylim([-2, 8])
axs.legend(loc='upper right')

plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/57e5d13dcf523d152efe6fbc10a6817b98341134.png)
:::
:::

::: {.cell .markdown id="pGtXGRoOWL5a"}
## Principle component analysis
:::

::: {.cell .code colab="{\"height\":295,\"base_uri\":\"https://localhost:8080/\"}" id="XsXVlnNtWLSK" outputId="24f780cd-70ae-405a-cc7d-48ed993f7a4e"}
``` {.python}
# principle component analysis
variance = []
dimensions = range(1, len(adcpdata_scaled.columns)+1)

# test different number of principal components
for i in dimensions:
  pca = PCA(n_components=i)
  adcpdata_pca = pca.fit_transform(adcpdata_scaled)
  variance.append(sum(pca.explained_variance_ratio_))

# total variance in percentages
variance = [100 * value for value in variance]

# variance bar plot
plt.bar(dimensions, variance)
plt.xlabel("Number of components")
plt.ylabel("Explained varience [%]")
plt.title("Principle component analysis")
plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/d9a146b9b4d71a8fa6c71ab5087ee3c080728d29.png)
:::
:::

::: {.cell .markdown id="hpGfrZWzX5SA"}
Select 3 principal components.
:::

::: {.cell .code id="MXibSsyQX8qS"}
``` {.python}
# select 3 principal components
pca = PCA(n_components=3)
adcpdata_pca = pca.fit_transform(adcpdata_scaled)

# k-means clustering, identify the best k
# with elbow method and silhouette method
clusters = range(2,31)
kmeans = []
labels = []
inertia = []
silhouette = []
cluster_delta = []

for k in clusters:
  kmeans.append(KMeans(n_clusters=k).fit(adcpdata_pca))
  labels.append(kmeans[k-2].fit_predict(adcpdata_pca))
  inertia.append(kmeans[k-2].inertia_)
  silhouette.append(metrics.silhouette_score(adcpdata_pca, kmeans[k-2].labels_))
```
:::

::: {.cell .markdown id="E-9tFsVDa3a3"}
Plot Silhouette score
:::

::: {.cell .code colab="{\"height\":312,\"base_uri\":\"https://localhost:8080/\"}" id="cFlC2He_a_XO" outputId="57344659-d1fe-482e-b227-58fb849e1848"}
``` {.python}
# plot silhouette score
plt.plot(clusters, silhouette, '-o')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette score')
plt.title('Silhouette Method')
```

::: {.output .execute_result execution_count="68"}
    Text(0.5, 1.0, 'Silhouette Method')
:::

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/c9d7db1afd86a427cf6e50c5eda6fbb670cd1337.png)
:::
:::

::: {.cell .markdown id="b9WXxDSvbI92"}
Elbow method
:::

::: {.cell .code colab="{\"height\":312,\"base_uri\":\"https://localhost:8080/\"}" id="V3OzwvoIbLl4" outputId="7dddaf23-13c2-4388-f182-f6f7fa9ed361"}
``` {.python}
# plot elbow method
plt.plot(clusters, inertia, '-o')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
```

::: {.output .execute_result execution_count="69"}
    Text(0.5, 1.0, 'Elbow Method')
:::

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/9d59bf6592a37064a31991aa3edb2d983e4e5a19.png)
:::
:::

::: {.cell .markdown id="oFpSwkEQbPI6"}
Inertia increace
:::

::: {.cell .code colab="{\"height\":312,\"base_uri\":\"https://localhost:8080/\"}" id="rt26g4UMbSX1" outputId="77ef5efe-d6b5-4f77-b18e-c2958157dfd9"}
``` {.python}
# plot inertia change at each step
plt.bar(np.delete(clusters, 0), -np.diff(inertia))
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia change')
plt.title('Elbow Method')
```

::: {.output .execute_result execution_count="70"}
    Text(0.5, 1.0, 'Elbow Method')
:::

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/1e90985660d181449ff7163b634f5f6eabd39ada.png)
:::
:::

::: {.cell .markdown id="AH1UoxW3bjo1"}
Select 10 clusters and convert to pandas dataframe:
:::

::: {.cell .code colab="{\"height\":424,\"base_uri\":\"https://localhost:8080/\"}" id="gCCCR52jbl_Q" outputId="e972d3af-2bac-4363-c051-874e577f22c3"}
``` {.python}
# selected 10 clusters and create corresponding pandas dataframe
clusters_pca = 10
colormap = labels[clusters_pca-2]+1
adcpdata_pca = pd.DataFrame(adcpdata_pca, columns = ['PC1','PC2','PC3'])    
adcpdata_pca['Cluster'] = colormap

# display data in terms of principal components (PC)
display(adcpdata_pca)
```

::: {.output .display_data}
```{=html}
  <div id="df-bcf99164-c377-4886-8c20-f4bfd7afb117">
    <div class="colab-df-container">
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
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.542901</td>
      <td>0.170293</td>
      <td>-1.957296</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.251878</td>
      <td>0.075295</td>
      <td>-2.576086</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.188105</td>
      <td>0.063022</td>
      <td>-2.123341</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.165864</td>
      <td>-0.039788</td>
      <td>-2.223830</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.684078</td>
      <td>-0.235824</td>
      <td>-2.242043</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1157</th>
      <td>-1.300526</td>
      <td>-0.932942</td>
      <td>-0.389172</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1158</th>
      <td>-1.152360</td>
      <td>-1.046499</td>
      <td>-0.283786</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>-1.028734</td>
      <td>-1.102536</td>
      <td>-0.288122</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1160</th>
      <td>-0.946982</td>
      <td>-0.972762</td>
      <td>-0.161716</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>-0.852845</td>
      <td>-1.023491</td>
      <td>-0.148515</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>1162 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bcf99164-c377-4886-8c20-f4bfd7afb117')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bcf99164-c377-4886-8c20-f4bfd7afb117 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bcf99164-c377-4886-8c20-f4bfd7afb117');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="f2PYBdtRdl5x"}
Interactive plot:
:::

::: {.cell .code colab="{\"height\":717,\"base_uri\":\"https://localhost:8080/\"}" id="xJP1udpGdy1u" outputId="f0788dec-fd67-4857-956b-8bc0701c7e40"}
``` {.python}
fig = px.scatter_3d(adcpdata_pca, x='PC1', y='PC2', z='PC3', color='Cluster', width=1000, height=700)
fig.show()
```

::: {.output .display_data}
```{=html}
<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="970e7d10-ad81-4b7d-8c42-28aa141cd4a7" class="plotly-graph-div" style="height:700px; width:1000px;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("970e7d10-ad81-4b7d-8c42-28aa141cd4a7")) {                    Plotly.newPlot(                        "970e7d10-ad81-4b7d-8c42-28aa141cd4a7",                        [{"hovertemplate":"PC1=%{x}<br>PC2=%{y}<br>PC3=%{z}<br>Cluster=%{marker.color}<extra></extra>","legendgroup":"","marker":{"color":[9,9,9,7,7,7,7,7,7,7,7,7,7,7,7,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,9,9,9,9,9,9,9,9,9,9,9,9,9,7,7,7,7,7,7,7,7,7,7,7,7,7,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,7,4,4,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,7,7,7,7,7,7,7,7,7,7,7,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,10,2,2,2,2,1,1,9,9,1,9,9,1,9,1,1,1,1,2,2,6,6,6,6,6,6,7,7,7,7,7,6,6,6,2,4,4,4,3,4,3,3,4,3,3,3,4,3,3,3,3,3,3,10,10,10,10,10,10,10,10,9,1,1,9,2,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,4,4,4,4,8,8,8,8,8,4,4,4,3,4,4,4,4,2,7,2,2,2,10,10,10,10,10,9,9,1,9,9,1,2,9,2,2,2,2,2,6,6,6,6,6,6,6,6,6,7,4,4,3,4,3,4,4,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,9,9,9,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,8,2,2,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,6,2,2,2,2,6,2,2,8,8,8,8,8,8,8,8,8,8,8,8,8,3,3,2,2,2,2,2,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,6,6,6,6,6,2,2,2,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,6,6,6,6,6,6,6,6,5,6,6,6,2,6,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,10,8,10,8,10,10,1,1,2,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,3,3,2,2,2,10,10,10,10,10,10,2,2,10,2,2,2,6,2,2,7,6,6,6,7,3,3,4,4,4,3,4,3,3,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,3,4,4,3,3,3,3,3,3,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,10,10,10,10,8,10,10,10,10,10,10,2,2,2,2,2,2,2,2,2,6,2,7,7,7,4,3,3,3,3,8,8,3,3,4,3,3,3,3,3,3,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,1,1,1,1,1,8,8,8,2,8,2,2,2,8,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,6,2,6,6,6,2,2,2,2,2,2,2,3,3,3,3,10,3,3,3,3,3,3,3,10,10,10,10,10,10,10,10,2,2,2,2,2,2,10,10,10,1,10,10,10,10,1,1,1,6,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,8,10,8,8,10,10,10,10,10,10,10,8,10,10,10,10,10,10,10,10,10,10,1,9,2,2,2,2,2,6,6,6,6,6,2,2,2,6,2,2,2,10,2,2,2,3,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,9,10,10,10,9,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,2,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,10,10,9,9,9,9,9,9,9,9,9,9,9,1,9,1,9,9,1,1,9,1,1,1,9,9,9,9,9,9,9,9,2,9,9,10,9,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,1,9,9,9,9,9,9,9,9,9,9,9,9,2,9,9,9,10,2,2,10,10,10,10,10,8,10,10,10,10,10,10,10,10,10,10,10,10,4,10,9,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,4,10,10,10,10,10,10,10,10,10,10,10,10,10,9,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],"coloraxis":"coloraxis","symbol":"circle"},"mode":"markers","name":"","scene":"scene","showlegend":false,"x":[-0.5429011085663761,-0.25187750567540707,-0.18810495378871567,0.16586352783248537,0.6840783058093939,0.611797865146219,0.49185358649374733,1.7203690011125061,3.245481236750183,3.6029403744673263,4.046987532471054,3.091834266569131,2.0797916762092528,1.7785543706960758,1.351256552418917,0.9717082248497512,0.33212037263797417,-0.2660555213963062,-0.686261108848751,-0.9984108248044287,-0.9320564398247538,-1.067660692771196,-1.0935005809453138,-1.075716300342488,-1.0197282807716153,-0.7700653883010888,-0.8204707518186183,-0.6262556797642369,-0.16499633405092756,-0.31521781344486666,-0.33918599051484777,-0.18978572309905545,0.24962122450646293,0.16004856096505,0.048608565179611235,0.02569565624773995,0.04422471438255281,-0.28179604168332595,-0.32730605308873945,-0.3187098088606333,-0.7839254520512683,-0.7012959448010702,-0.8908266430911994,-0.798613844244073,-0.9427330050948672,-0.8385418606872613,-0.8784888461390696,-0.5582598339689702,-0.5085991888663717,-0.7190804968211014,-0.05555979480988395,0.5443315621848733,0.8728222041586762,0.8027986406876574,1.1723429856373924,1.3302181189568423,1.7022291741728595,1.5643817639445672,1.4249236863861603,1.6046012585327942,1.6710283997103414,1.471026685595088,1.153091280743586,0.701865490482283,0.5657175522836985,-0.13354174250157005,-0.249421491600918,-0.14792565193843876,-0.8333744015851952,-0.774301159791434,-0.9296339391582187,-0.9650792637749059,-1.0462746627810056,-1.0142014973725837,-0.8389779469124915,-0.8882538003543816,-0.9690213441643406,-0.8927687974703,-0.6019865968903461,-0.6087259186114753,-0.6705158792677214,-0.24041611824583997,0.04134245247241097,0.26492169509760816,0.5213023765714925,0.2495032902603602,0.336985133780439,-0.16020034151286747,-0.4066997363827331,0.08678820272557261,-0.3188759368080814,-0.5397900762481661,-0.4263484490644314,-1.0120516044931596,-0.9340307092824586,-1.1495233621900935,-1.0816121341172462,-1.091793523543806,-1.015923946684035,-0.9858635352785711,-0.5459104848252818,0.04316458475774904,0.0709140210457088,0.11805096512315295,0.6800114266998627,0.5222126451307786,1.2738891228443456,0.9229737304067355,2.5197725457806808,2.54414511897824,1.2699281189920248,2.8539750789822116,1.60118507175134,1.5428603287630953,0.7774388237941148,-0.036499951563885635,-0.1386515090403698,-0.26872082061031183,-0.8200716367320547,-1.0188798908796228,-1.01444056278277,-1.0198150093975547,-1.0357535358667476,-0.8705181573142722,-0.9838638609467643,-0.6784941049280472,-0.6970486227060272,-0.3450336133216887,0.0747583489048162,0.38041110517363386,0.03289706912695368,0.7433160946994474,0.5835898564074483,0.34641648543173903,0.41217805629906235,-0.2724637201517073,-0.4450601277085523,-0.8644632597263772,-0.8550717559749142,-0.9838394343099794,-1.2890755305560144,-1.3562198004239665,-1.227437953369266,-1.3310422502585513,-1.117566476026354,-0.7541163382737089,-0.14530227232532872,0.5360560784619299,1.4300460566043487,1.0374180760382117,1.8663908250909123,1.8599898256079377,2.029872039304777,2.3500603421410604,3.1516948852470086,3.343357139580235,3.1387493633016663,1.6127952659272211,2.3312663073952384,1.6742604452184238,2.186820424540895,3.308319377388533,3.506622904182328,2.1671807316820413,1.1738799968214764,0.6724520439018868,0.6128596421684392,-0.09099108151352188,0.1934847305839931,0.5187879146911768,0.5069570647350954,0.5123573009823712,0.5169219388769459,0.18867402042990014,0.0163852233603478,-0.49608010891335524,-0.22544187220319936,-0.2653453942016289,0.1437956202544048,0.23637881478699393,0.001500928785240245,0.03875863338458674,0.16518235558894112,0.10085540717752597,-0.2355244985294888,-0.24422302082899439,-0.44105488879755234,-0.35313865764803276,-0.7404438981253569,-0.61923917363573,-0.7428224389197423,-0.6834312791867931,-1.1598228306834928,-0.5796929477309432,-0.6915355380521107,1.1056245970938174,3.164314267542505,2.9954106804264917,3.260958834114255,2.817192366442625,4.776956199465978,3.6736556293768143,3.096093152907121,2.1729000000732244,1.6771944996782457,1.5247708875417239,1.9808122769792922,1.6285502568193648,2.421874023649143,1.984651001579129,2.179070668621472,1.3878626898050255,1.0017557896844564,0.5395958639194184,-0.08702132228619155,-0.45696823954672205,-0.3642036260463535,-0.5738914867609473,-0.7698446928570998,-0.9291669712023196,-0.8993438927108848,-0.9258852874421554,-0.9009110462153688,-0.8827489377316902,-0.6957359869707355,-0.34305119548201213,-0.2893344543196307,0.12192524734207644,0.22748701176561043,0.5658545474840828,1.4701415119398205,0.857638072483335,0.4953588464281823,0.5110431465097928,0.7158723258678146,-0.1562354020696172,-0.37130656109086824,-0.34601238345839125,-0.8199608564998727,-0.8442907756530302,-0.9347201628821789,-0.7313123343441088,-0.7039851245432566,-0.9308855914678463,-0.7003283497448329,-0.7400889774973998,0.10394205194865924,-0.024249190804614268,0.6863823778399446,0.9339422020772739,1.1567849225920752,1.2981251748753926,1.3085997207984832,3.124786269829427,4.240476937903476,4.000112742796038,4.606841830972913,3.508972051679076,4.1463997691214765,3.43681345013749,2.4010654340086575,2.8284486724651963,1.764474587033804,0.8996280408958696,0.055085879675757864,0.3153096878387127,-0.05237676468415638,-0.331793104816783,-0.45652059538781165,-0.16963071339431385,-0.6580976045079123,-0.6640411736723111,-0.8198153427995325,-0.8132082130630813,-0.6525066230680299,-0.8395156042072798,-0.5312185362093788,0.6488239711518586,0.41415949597415136,0.7242352803027243,0.9982432484578383,1.3726873985555923,1.5964932886773098,0.7571188239261748,1.3502263029896202,0.4687765082371868,-0.043519677916622666,0.2104331046245847,0.0729296616848776,-0.6536278225713077,-0.3999491137323421,-0.11021962691493432,-0.8854569632703166,-1.1070060995600965,-1.1359676716040996,-1.0037887269190016,-1.190762059315351,-1.3796638593586201,-1.2317242336604999,-1.1828865906922026,-1.343770706658145,-1.339533561729501,-0.8680561702062317,-0.2798411663919659,-0.3431752630325676,0.4507736437630094,1.4838502183564588,1.8647586305882038,1.7243525294276987,1.453802854501546,-0.12647228947162048,0.5781477116188528,1.090916277108495,0.4304811112866663,0.27363810264005206,-0.13710760480057194,-0.6382772386421681,-0.7936148198023995,-1.0680887887251933,-1.33820906143279,-1.2217959954298543,-1.2771043867164875,-1.032910159406415,-1.3598710752098182,-1.195502218195193,-0.8896625086776402,-1.108113245933614,-1.2147332535072741,-0.9013126026312958,-0.998171916327705,-0.883688956777091,-0.7300788212185066,-1.0477266565010195,-1.4920839928203682,-1.1147664367013266,-1.317956756135604,-1.627834053979207,-1.6391351304926522,-1.6647237447940995,-1.6982624948382836,-1.6810010368896624,-1.3901665826218945,-1.3574658857111987,-1.3499748158400402,-1.3731913198971601,-0.7927368444943991,-0.042212166020590594,-0.11534306609876664,-0.05152708038149361,-0.8468448021568609,-0.48883907922756004,-0.3855011824007434,0.3016878368867769,-0.27888046796178834,1.245362319533023,2.4860077309553885,1.581681442791564,0.4375102968150753,1.8055963231825396,1.1094523180606095,2.2283541306761285,0.6040092374875672,0.5595207805854806,-0.6163970091948906,-0.23963913133271122,-0.8978393967949687,-1.1901461859184963,-1.0729513928712284,-1.261292277064967,-1.0918134950136509,-0.8925176876869142,-0.9129000135889459,-0.9322782684217936,-0.669221249824819,-0.7358604993679135,-0.018891867399515278,0.4231671382926734,0.3664141638824609,0.8790467874567954,0.7060800453741182,0.2993435355095633,0.16661502305458065,0.26964381901781564,-0.49632641742357686,-0.5541070006485883,-0.8434490640451003,-1.2403494880246546,-1.3951128281878598,-1.4281379012375381,-1.4883546943935253,-1.4665813650360289,-1.4905546863148336,-1.1365972044771047,-1.1638493807810235,-0.5554373568077651,-0.8819936179791175,-0.5502639738740726,-0.3279893279963028,-0.42266189131010184,-0.49344289421021753,-0.09764076998183809,0.12041104251537528,0.906242568966354,0.44245565808645915,0.7362297211589792,2.3832598023370704,3.134630661773592,3.472275594734115,2.593587871765663,3.0577245548552003,0.8778073420137134,1.3144899884901695,0.5511558761270837,0.372980093100148,-0.9073858650850362,-0.9490874809966663,-0.790060578678991,-1.1778694306995752,-1.0841189464337297,-1.0601393235846754,-0.8118528452820899,-0.8806507377360343,-0.9284724612728306,-0.8947743476377994,-0.5175177888601539,-0.23933854117020426,-0.2523194333299383,-0.16614503404297068,-0.27690307949966236,-0.12050635811097599,-0.44729637427508756,-0.5507454897415552,-0.5703466783986471,-0.6997786238907954,-0.7134109353155063,-0.9588136545474637,-0.9518834084217382,-1.0376996604012225,-0.6136958194485652,-0.9959359337639744,-1.0837793651876226,-1.152005998948876,-1.2904253889869797,-0.3722738132564327,-0.37166572151083765,1.603306985640763,1.6499815821987076,2.395822758964835,4.551599035440408,2.4389852771566862,3.7390975545403946,4.767139226857631,4.854377517440472,3.790016579941965,4.09446937353151,7.26403736456505,4.442146970800677,3.7988975405893117,3.462729818778913,1.7224676937367491,2.706125895685746,2.6073077632887296,0.764001484614032,1.0723297170031576,0.6753082587667365,0.5362679568365977,0.26562206989706694,1.0586384309230585,1.6229257468788647,1.1770337967708446,1.2107827770181665,0.44554065964344325,0.18988907109973496,1.0249363304520651,0.5089669057502717,0.9405522609041619,0.6348771472529524,0.7168487681298898,1.0785354457932852,-0.010359358577826893,-0.10718920982592582,-0.2403792666693451,-0.31520751125591434,-0.5442175222546565,-0.4322770222981955,-0.19236801139042384,-0.6300796852582905,-0.1754435105150434,-0.18500262437183024,1.171889089672343,2.5978802669496126,2.6570347009795503,2.619261422303224,3.0961723935364325,3.283234100059269,4.752252086173785,9.004227439757695,8.357051602954657,8.956870076782469,6.538305619697198,6.989079843799436,6.395188462430617,6.319614598790417,8.35886212955106,5.440415015884828,4.610930685963394,3.051618116512449,4.244069022659655,2.9296086254711566,3.4508381927411778,3.097981908299824,2.4236503076448788,2.2717613097856924,0.5727487584267819,0.6631503065249741,1.0254579764020042,0.7576282502919265,0.84512923464951,1.0308624920022054,1.2893577935067002,0.6047043570612628,0.3677578367807108,0.5303644360306904,0.6408998752207383,0.5957833369678543,0.7319037598585877,0.783756504725384,1.0489848430444775,1.346290609902237,0.9093207115491639,1.260910287507741,0.8688532511866842,0.5521301573395995,0.9863988596802984,0.7217984643927329,0.461015025577676,-0.045139371799819444,-0.202745842137657,-0.29117196207094076,-0.3767867594797101,-0.5425826997216441,-0.4856664429898043,0.3606935954306672,0.0677575157874015,-0.055075033971474355,0.645431987965328,0.8664407569750002,0.6924565100130777,2.066737658463783,1.4953597621611079,1.2641552066700346,2.318336337793782,2.813510497650324,3.2587522556467676,3.087363308361383,2.1102710877759745,1.1966884948655818,1.6546560889516861,1.0662343531496605,1.1340994705076262,0.9891891919776704,0.42707028851993806,0.16908338253511587,0.0764704148656188,-0.2344340587687549,-0.21889235772610788,-0.26602150254015644,-0.2754467989485416,-0.39235425131744084,0.037100266998255534,-0.13489398504346803,-0.10968819616840038,-0.1352113890336685,-0.13596678171354742,0.5678816911207776,0.9862312323840504,0.5936456260540681,0.9780019551108203,0.6430662848484648,0.24591474750004644,0.05626487934751592,-0.25537324696523767,-0.16181114491370938,-0.3542457915203418,-0.5014999912749505,-0.4747470423990338,-0.5653996135036965,-0.41243101867065324,-0.5803027543875042,-0.7953771897665166,-0.7905422433164548,-0.7738410124617223,-0.8396630095076345,-0.9200894270968416,-0.9451236374833242,-0.9351116223174346,-0.9745884942730829,-0.9569378103509125,-0.8274979158330346,-0.8338310098458375,-0.7297088611834852,-0.6179641566988321,-0.44244373330794273,-0.4221384902741954,-0.05970054762520581,-0.28995502376688176,-0.1384342443274454,0.24499505956142903,0.2539581180909061,1.0071771199018427,0.7421051575995858,0.6100858214132916,0.8226851409228406,0.7074278538317224,0.6968057375143801,0.3636308002470038,0.39657182668679825,0.230281560454729,0.13085273332891903,0.13329595596054428,0.006241074392974347,-0.043792385162341754,-0.022867106871181604,-0.16124882545886765,0.08760241251557216,-0.10618844959488698,0.11041873510929004,0.19055698724608702,0.28847329566402957,0.40954237715449393,0.1617722594934702,0.014606542854363863,0.4562612236859999,-0.41076781873555146,-0.17086621551328401,-0.5646602091598063,-0.4598269683315508,-0.5542530082553344,-0.398482221816126,-0.7594388849070548,-0.8538307137835489,-0.8345050528439314,-0.8197727728927496,-0.6192056406739176,0.22998445358803513,0.006245160382360241,0.25445425557299245,0.8504833396445398,0.8530055216529742,1.3154487638800447,1.7634143114783505,1.2752266704819653,1.1991801566389815,2.4609151458217373,1.3949384210889897,1.6400861370642579,2.7965943457997087,1.3284231214547044,1.0963937530376942,0.7323485620780256,0.3193423859150531,0.18541956708945662,-0.18646830381818916,-0.1906165936127858,-0.08490614017679445,-0.4164055915002297,-0.6089409755105605,-0.46985637868574565,-0.3225688993188345,-0.487591423440479,-0.4563333343424522,-0.19153750291396754,-0.4163293664206701,-0.11576058670582394,0.11930429086505344,-0.17251288431701822,-0.11963266409396281,0.08576316780141377,0.5596843308660644,-0.4075087838414252,-0.1244812158160487,-0.10508358173997553,-0.6231143885588917,-0.48204526635214356,-0.600093776524288,-0.8844891370349071,-0.6808049890653001,-1.0107366567352727,-0.9629475118283837,-0.9612321591057982,-0.9672957666723812,-1.0693057461947426,-0.9291212988076163,-1.1787678074128118,-1.2065872877657413,-1.1406094677580183,-0.9123868948326306,-0.8632653935770621,-0.7341398385289317,-0.6580565282394787,-0.5792044209113185,0.07543565849183236,-0.1057354890479488,0.412136054792534,0.249138945550364,0.2939286648359775,0.001010276300748001,0.5599865044699799,0.2976292831716821,-0.7990612298946683,-0.7925270791756317,-1.0245377050532294,-0.8512028470727822,-0.8874192465480583,-0.7167612157077267,-1.0478586301693305,-0.8913597170587949,-1.1060943379762254,-0.3864518695894895,-0.6548757675478276,-0.6160793950132898,-0.39267256233442926,-0.2485601159998368,-0.4106936206050335,-0.1505113524366388,0.04280637586573279,-0.33694316519918266,-0.5119409837247548,-0.1499684121858451,-0.601549215221148,-0.8631278566939762,-0.9116944097478153,-0.823544052827316,-1.1064773201817943,-1.2469036671439584,-1.0133682580377852,-1.2105140734938833,-1.2698386227856413,-1.285885789301863,-1.2586120848912588,-0.7909324558690672,-0.5145684259963127,-0.17394477016597398,0.5596310157362966,1.1632539918443674,1.721552954823403,0.83434390260073,1.9307621718670103,0.9966662583477408,2.2574465127673506,2.1225603249692266,2.3402746346123355,1.4660266783148954,1.5823454458371162,1.6222225126693581,1.506865125029655,0.36799991265255194,0.5143965121405057,0.5503981456062929,0.10879100643725675,0.5199490013139448,-0.5706149619392094,-0.3054529821365064,-1.0451840687656273,-0.6238421355098811,-0.16079866706608456,0.16318779606874204,-0.03878054962262117,0.13806133901530465,-0.1889071720892707,-0.5942357253207557,-0.9117400727086516,-0.8199588142835567,-1.0116355084762447,-0.7790135266119983,-0.3801671770702319,-0.4096226882014505,-0.37585143558358275,-0.42320860653014364,0.13526009394587493,0.4769303416241578,0.6707576033629202,0.09542850412721955,0.16448230696834726,0.01777762343700249,-0.6064404609093489,-0.309362269317547,-0.45897998594055994,-0.3287325804846319,-0.5892504551392188,-0.5333204548520591,-0.5227884625179029,-0.5970588399904317,-0.2569008907925662,-0.6161552391326388,-0.30119124390941293,2.325366228369194,1.2744366934341047,0.9570982732730766,0.5767714724720509,0.15698940563196753,0.5464342761798122,0.9590192693901638,1.0734227760370878,0.23701974306195134,0.9623107802309875,0.7598629071571131,0.25437147433764773,0.2653675157207302,0.2532733250627568,-0.46521707665982454,-0.5323739168222695,-0.48021525225280653,-0.6858153132174615,-1.0028184523927863,-0.9536921978539515,-0.7221161118506088,-1.1194644946964218,-0.8777475359646957,-1.1026861435461934,-1.1158304820932414,-1.0923967642130787,-1.1457905682301934,-1.148003187917429,-1.0915578477439305,-1.149820394646022,-1.0785956460555588,-1.228637283168344,-0.975856437772861,-0.8172638954928507,-0.8132544464216932,-1.0040628252953563,-0.6279203682932003,-0.4715313836572647,-0.6123933507139029,-0.6570781006822145,-0.48770540221868874,0.885005932434901,0.7278476908961758,0.8881970934216299,1.4797916590825846,1.8276928925581586,2.201012229075969,2.7288417057828855,2.8901568346527386,2.1164445372279057,2.5331684977978304,1.6697139053719934,1.4762615419543141,1.3309018848311902,2.142404261319014,1.4189700219704828,1.3763181649667302,1.417520989088,0.2553918724425981,0.6592821980841703,0.4313520880148714,0.5380556401474689,0.1499352458954047,-0.32256856521022176,-0.3840560730134549,-0.7499804759885076,-0.9945054336142438,-1.027811816890969,-1.1764521727367152,-1.1002993778644123,-1.2032282928644442,-1.1505723711760472,-1.1037110390366875,-1.1943633596783854,-1.1282124159813416,-0.977864350273614,-1.0518687089582324,-1.005053014746896,-1.0415540012030222,-0.6694278414425513,-0.6938665901683189,-0.8884396777619826,-0.6395030856851311,-0.43519162725769467,-1.2396516170075669,-0.7399873596874914,-1.2638869305278626,-1.3204892513036213,-1.2025657206128035,-1.4220401561437734,-1.5532537660939503,-1.4420102458141573,-1.5443383382839824,-1.5967080488173886,-1.4772440160536635,-1.6794942968702782,-1.6155840452562182,-1.5483383810228917,-1.3895116427955692,-1.427184648702527,-1.4415882416630794,-1.0368341825482794,-0.9519405957476182,-0.4561875248141752,-0.7010527338759531,-0.6340148256965012,-0.2751257833759322,-0.8422302238695879,-0.32641109451795397,0.0755850494536255,-0.11992927045808104,-0.43609654692765826,-1.0050774236598972,-0.9804589972129686,-0.9582718210038366,-0.9234387613329695,-0.8123608757605618,-1.403136182008481,-1.4172594490514907,-1.6815861107097223,-1.421689441922845,-1.3880541187533486,-1.229841555815442,-0.8322229415862609,-1.0985708016030118,-0.7602640542502056,-1.057738766774301,-0.7953557558985949,-1.1230786850174963,-0.8176279108498756,-0.8879240407087483,-0.855826507198083,-0.9456899891597591,-1.1381031643112896,-1.4840043723802137,-1.4995056026095388,-1.3388039114046786,-1.6411810143334933,-1.3416814884145634,-1.539880270384982,-1.3774562744862657,-1.5710440711755316,-1.632036625837981,-1.6143895738714082,-1.6355748463210407,-1.513353988879607,-1.3102948543057178,-1.342517359446184,-1.0598796530854846,-0.6961109839610443,-0.4572651109479897,0.377456149302703,0.11075511437891049,1.4095430070517425,0.3597966984756719,0.5144405603254087,0.23634804076883761,0.16685548670291203,-0.02332991034773602,-0.13123639573667126,-0.424370464173659,-0.6516892035572055,-0.8407129498095278,-1.1246707108325635,-1.5070264944233573,-1.3132598159665638,-1.4732083067728854,-1.6664247840120647,-1.5348410299716504,-1.5723728605419731,-1.2082609409151108,-1.4634073819923759,-1.2413122226772457,-0.9352251266275187,-1.2169590283750986,-1.115318223458476,-0.7286646554130698,-1.1165425587043138,-0.5733540836024504,-0.9535019159162997,-0.6896463604760307,-0.723415986239028,-0.8257198847238495,-0.3338264171463427,-1.4586890272627444,-1.2123428468876658,-1.0055822734300413,-1.4554588864894242,-1.4798455765299705,-1.7242998671777618,-1.4359201732355278,-1.3328001496252309,-1.5550534457598406,-1.6127742417374862,-1.6085863430981797,-1.5481922800515164,-1.6416390151435134,-1.6276922124217488,-1.3350247210538697,-1.3913327154631578,-0.9245811025118631,-0.9286097236252029,-0.8715827878046364,-0.45640190048983437,0.13525330158051935,-0.08292047652532313,-0.7079188160433552,-0.5874058341206441,-0.2622786011042361,0.826890282086801,0.0539448734108285,-0.08131651191565387,-0.47453172616288813,-0.9937857327423834,-1.2266151260094758,-0.9341271501751136,-1.50610523376563,-1.4796467615036326,-1.437792974214838,-1.465127803443861,-1.149512504245126,-1.021538394990931,-1.1905307311271265,-1.1436078101432292,-0.9925121404791589,-0.6210445617258902,-0.9039308315119493,-0.6908135546738201,-0.6584022796979808,-0.931093186701952,-0.9259684911530534,-1.0222519595265331,-1.3946146552402874,-1.3000925353061004,-1.2194403387794355,-1.3560389434495785,-1.3784168248359394,-1.5895490706261153,-1.5114799398892638,-1.3154299814326305,-1.2944071222904505,-0.9005160127170675,-1.0547076863603329,-0.3813000338750167,-0.5361044596421193,0.1219917920838507,-0.20660674555306305,0.331617689329071,0.28429135340466183,-0.3454018957332383,0.1611970640049933,0.6048793286694046,-0.0009995230450285662,0.440545816453294,0.07242931463936136,-0.10067168356588188,-0.19947491943051662,-0.27982444498224956,-0.008412296823165681,0.1833418715462254,-0.2411601718662291,-0.691354049418876,-0.4211605695624082,-0.591754338852334,-1.078422123242622,-1.0578395561148426,-1.2226141965390487,-1.2472774933273783,-1.2670568479091362,-1.306742676378772,-1.1730917311744993,-1.1150480539111582,-0.840298071231555,-0.8277930129791874,-0.9112768145895079,-0.6442056861299192,-0.26428913138664173,0.19860945593662443,0.02763799407992161,0.07951219971707853,0.21424878548041928,-0.14222462389205864,-0.5586389281063373,0.31328660217406107,-0.6883701132501245,-0.6332279415312823,-0.8021226212532092,-0.9190673691411181,-1.0766075367270944,-1.2642352719418914,-1.5104756116925537,-1.4536746405982803,-1.4313971105757342,-1.580978688947783,-1.394285004707509,-1.3449272424653664,-1.3420021772563786,-1.0743236046470013,-1.1115359292453055,-1.1271717521323157,-0.9423415920825777,-1.2773325463202747,-0.6872355752816629,-0.2892520527319313,-0.6260616592324983,0.042342027734871286,-0.3566338200447019,-0.18091718182809044,-0.28791915938622015,0.08513201610628825,-0.03907469428459601,0.2261902579819515,-0.17627073180156388,-0.28202635617383665,-0.27209463153230934,-0.855795174046852,-0.5744731263294393,-0.7508522679889402,-1.2790272149905706,-1.2080390182315925,-1.2070984339020252,-1.254562080642041,-1.158310502090249,-1.3230110487815983,-1.352562395417471,-1.3048043608734714,-1.4016229425920554,-1.293888699002834,-1.0830554281808997,-1.1147799119100283,-1.0453419693143968,-1.1105541458127235,-1.1395559487421227,-1.2703543752011297,-1.4305560622031839,-1.3767032020123702,-1.3573965341227547,-1.392028663167899,-1.4470625120746154,-1.3781534120650931,-1.4641872605795618,-1.3199946182858848,-0.8469753690251086,-0.9472672420664354,-0.8375019063637891,-0.7910088773250368,-1.0144440826939727,-0.8861921674053778,-0.8134329675883316,-0.8996370242900702,-0.7589135501506379,-0.9916821755656307,-1.120331361497143,-0.8717760799403144,-1.155474922138149,-1.0463086377086404,-1.1567367854261899,-1.163939970203941,-1.301345616121972,-1.348196036652725,-1.232811344739853,-1.3005261653241755,-1.1523604825655047,-1.0287339879907385,-0.9469821986321418,-0.8528448734454703],"y":[0.17029314639954155,0.07529537068971882,0.06302222929955163,-0.03978840547486806,-0.23582402649626,-0.2737671423426207,-0.47419031143645296,-0.2515098781898194,-0.3725040834807578,-0.3751382252534189,-0.40323315836072027,-0.6716936132023369,-1.0565376359584069,-1.225087659120591,-1.593272701928097,-1.7747520741299299,-1.8182319390325248,-1.8723442170256095,-2.161469968988785,-2.0162160360089567,-2.2970502135267945,-2.119606197734097,-2.059170954878393,-2.161893765724395,-1.7070582515463015,-1.7067622054300877,-1.8310089113066075,-1.4234894327443974,-1.4294957782926203,-1.1526813044676605,-1.5001981970880893,-1.1989591946068783,-0.7128489431449261,-0.5812981133435098,-0.599187520608482,-0.44220478749412473,-0.46260401115256256,-0.10545714459065077,-0.024999673184846062,0.09417005762668841,0.274913831429173,0.0351033796014496,0.24752082045906787,0.21828854144944165,0.22929758483216744,0.20569573551214967,0.00343681463260116,0.29266750369555294,0.2013140037099059,0.0878834861994871,0.05734565673194078,0.3358194929043263,0.14359724215067327,0.09753408498303802,-0.03260964156249099,0.06981977930379175,-0.2066232371926559,-0.24466887232413317,-0.5101530828285777,-0.6104614754418851,-0.7420392624900563,-1.152840807128596,-1.2651393798382842,-1.335988105652915,-1.6717634698458301,-1.9703391911376613,-2.113846092535392,-2.1624134064639997,-2.274416628233912,-2.357451653457247,-2.404946196916181,-2.3657263698120676,-2.4365187024212442,-1.9803942127472953,-2.032485474169876,-2.094903724606511,-1.8092334543171542,-2.0705913008660826,-1.961273087543347,-1.5796868840258713,-1.5054889079244356,-1.357913315935992,-1.2305163769257463,-0.9871475036087699,-0.7833275006638619,-0.6137113248198135,-0.34701732868587215,-0.09485757203583359,-0.1417624165478365,0.06170591109413054,0.2530784920444983,0.4604585610887088,0.21513913161977133,0.13905227497352907,0.24436304459578542,0.20502777987874984,0.28669363047061713,0.3559319619625111,0.37612524240815903,0.28128578525409137,0.29127081827485785,0.19570794807083308,0.1318052431234215,0.19714602915860663,0.08516679064425998,-0.21324162608546776,-0.07968478998636366,-0.2843974861278373,-0.35103270921718344,-0.5030483919118865,-0.7496504012956897,-0.8486295802273218,-1.0661419359821547,-1.1955067431875663,-1.5438839397617787,-1.4975872460388688,-1.7300009420651108,-1.8061823125412355,-1.667162529694886,-1.9596036335350189,-1.7805865530133809,-1.813290452995977,-1.6665163501223437,-1.6957253996644028,-1.4602355434563907,-1.398229749737706,-1.2291602588698964,-1.416741347514344,-1.1580363719297324,-0.892913502460857,-0.3376459340164921,-0.04619243690787101,0.13299104381850857,0.18530290637357955,0.32871738252679117,1.1349297742512086,1.3720702268153628,0.876838774393398,0.8642180710246431,1.4788363051192162,1.165289151410107,1.1525024170279876,1.794205658113003,1.2812480748016746,1.96053966281195,2.195371274064648,2.365321067126374,2.61497492110278,2.265186196805102,1.9305038592103723,2.5478150930993873,2.2168763452719005,2.013135828800624,1.6678971793009028,1.6542968727671588,1.0424004352052336,0.5529156502085678,0.1654495496788092,0.1838432140227893,-0.1872147995977981,-0.33435230119922354,0.3382570529260614,0.19255548670615621,-0.22503931095077417,-0.4346314953714377,-1.2787856008805571,-1.6165033969361815,-1.8264446184950396,-1.0920926890266487,-1.9613952232736307,-1.678581966306144,-1.5744985578970863,-1.967563411744657,-1.7147047538059397,-1.377934481120873,-1.5427076970554354,-1.7029671589491204,-1.2362177383222817,-1.246805396432976,-0.9749725905743367,-0.9616766312423263,-0.6502138692335566,-0.5973240402952856,-0.3699313688999392,-0.1476111239369757,-0.11224203993149241,0.15105132261776824,0.33176132239763256,0.5742852282052949,0.5218558883784786,0.6910537609190476,0.9426370020222138,0.9169077129813279,1.0638867477119494,0.8399888344469524,1.058473061915823,1.362094019034413,1.3379442602605822,1.4287032069701917,1.3794709739136684,0.7809566919145177,1.2087592876257602,0.9436810989595363,0.800246826462809,-0.0006915820313006833,-0.3503613569114458,-0.30463279749714073,-0.5527464855697978,-0.61141253695129,-0.7465656400410491,-1.1674676028371118,-1.2440760836711902,-1.4860349059620288,-1.3955704707738097,-1.5673343378324367,-1.7454747729200815,-1.731677753666451,-0.4672036196460811,-0.5247826640979267,-0.6903742938143685,-0.7479744501941659,-1.411316001476165,-1.666895771915522,-1.5978010034070962,-1.610929201894174,-1.5083306028591035,-1.3762926919177345,-1.263459309411236,-1.216768389099966,-1.072179655985954,-0.5222872892061885,-0.7249732366389869,-0.4890708338277592,-0.2883814305849508,-0.00760006632474043,0.06147779446388123,0.311922274019269,0.13135695980927356,0.2842383131564421,0.4500911221540198,0.5545142369077855,0.7243229901595537,0.8550764488532321,0.6046601298672829,0.966836190792708,1.015422200001444,0.9939935657588153,1.1088775837864435,0.965111516456743,0.9273648996805158,1.0026253207193632,0.8311772247004761,0.5836984735093526,0.6871369427353439,0.7591000317680554,0.5149231236468698,0.5424310862038705,0.21383964444967982,0.26956035666729944,-0.11250433350178081,-0.510515717468674,-0.9050855114253255,-1.2743367265364607,-1.548108490248861,-1.836642882471356,-1.4542302772334468,-2.0045003237299057,-1.812511076006157,-2.2551307226071833,-2.016383675838498,-1.8632200233080585,-1.8017164353934279,-1.8532083664485235,-1.7140749330867586,-1.6983421433291876,-1.6581639538458042,-1.5837836691472862,-1.3405289838743462,-1.1341875883661454,-1.0623559254390547,-0.8393708043879828,-0.34930717987073734,-0.19219269053988752,0.20770675601246305,0.2681078638799594,0.38633322183374913,0.38418136870389147,0.6805766794464738,0.7452783888721153,0.9270242354248485,0.9467903568769381,1.0032838426387758,1.2587775235196832,1.3226764143723784,1.6923811781112832,1.5851119622861447,1.766255703625672,1.6435207340753064,1.7242699627184972,1.9336418861799731,1.8570655619174516,1.7699906202003006,1.3341583126559442,1.3429768419543828,1.1744879788622848,1.1324682467877472,0.7722746299190486,0.8292348493869084,0.7136803818361184,0.30802121895699863,0.3166035796428136,-0.01566014812247564,0.08249333020979062,-0.19753828746262095,-0.13192218522452173,-0.1972678588977737,-0.3505810624675617,-0.5278268521931042,-0.671706625616037,-0.719843489405907,-0.7061442516063481,-0.6595167177483918,-0.6319000197249556,-0.35672151899445814,-0.38438631092575337,-0.30222885449730413,-0.1423235704352054,0.3183165501209784,0.21255701504248553,0.21986420966236295,0.7507668836543948,1.1577133299297284,1.3151111498295414,1.5170761667842025,1.6191599025981098,1.6681897189105834,2.4382661408077997,2.2300785876517852,2.1526427366696614,2.4466599015034807,2.598645136020054,2.2786583697727347,2.2470987354516616,2.221969514420613,2.837571537325826,2.937793772104655,2.8752289679030216,2.754278395486697,2.7904491371265716,2.904941586724039,2.8986816223152156,2.741739848996855,2.692967052203068,2.292146541488129,2.2735238122510233,2.5298521783107204,1.6650367553606344,1.4039689269111055,1.1021293549094762,1.163000821611555,0.8860556629416839,0.5634708399758672,0.41945163338656016,0.2888051571485575,-0.05980730549802025,-0.26524115258759046,-0.5887189949741912,-0.7739451615560354,-0.7757183467572157,-0.8453153667317262,-0.9637306297642156,-1.0745239921000713,-0.9446332938315719,-0.8500894197593314,-0.8368948571190729,-0.7167439711547169,-0.71388339743446,-0.5850208447590017,-0.3373239160501272,-0.13867554659836467,-0.14248779447765533,0.23810963584665956,0.22835354867107718,0.3274200757063946,0.827600795662082,1.0231708734440983,1.0004085857398493,1.3172127260591784,1.3463115089626245,1.481418094662618,1.6390797195913456,1.928233806923513,1.6680917543861389,1.866641906685618,1.8098952743812609,2.2159005284824014,1.7686967698895262,1.6306079277703545,1.7081219905059384,1.4963922748999654,1.4559318315646212,1.353712731210086,1.4686884004296392,0.9871806770953234,1.2019111297513345,1.217707672106509,1.0746787722107858,0.7123654250845162,0.6923908458567024,0.5404070763851349,-0.052070506198019464,-0.15176334451198495,-0.3052292720051388,-0.5613164630767218,-0.6863075011386142,-0.8978344822351152,-0.9239889585220856,-0.9910511290606288,-0.9416587658581623,-0.8609372432475991,-1.0671731325584797,-0.7395102148797529,-0.81019061916536,-0.6869237273230068,-0.6858494370418872,-0.609247096088822,-0.17696526073132834,-0.28757486584225384,0.025262026848091906,0.32376957704184095,0.5330682468823341,0.6248675665677907,0.7134352369160615,0.8782862778244915,1.128423212904105,1.170602252209771,1.2767011110570714,1.5103813014521401,1.5097661890077951,1.6198028824333186,1.543825881565152,1.5641582068248046,1.771222916585208,1.7048615577790824,1.8373741904948937,1.9843049704546598,2.105910841994289,2.090250269426092,2.352849569194638,1.5134078434620981,1.507930624210032,1.4967110342810863,1.2026346465255913,0.8358659529939904,1.040344697586535,0.83779260534521,0.16981563128413554,0.14325672332992345,-0.0641697914414686,-0.6311230262640879,-0.6608351309723192,-0.7381229574502056,-1.2894947440107636,-1.376278136753311,-1.453606055445559,-1.5210384249322042,-1.6620496029098986,-1.5168362962349424,-1.3638055048309383,-1.3683796191112347,-1.530763498388347,-1.816221797902183,-1.28842125428828,-1.3037066017461756,-1.3442432460393132,-1.385829006016403,-1.053875148159262,-0.7636722465715382,-0.6944854571596107,-0.7685553230101815,-0.5906425794971191,-0.473527850708082,-0.2770691653284345,-0.21450896609259054,-0.058003640596684045,0.27916374455827325,0.40088558416984055,0.7191822826026687,1.2638909842717154,1.0486330904917847,1.1107304647863288,1.0616081532829151,1.2936089876832177,1.4777784533310971,1.4686108901586563,2.103206174826285,1.5599476677300548,1.4718586180751805,1.3548141175540316,1.023641479686907,1.2707214593050564,0.8985956319080944,0.7323141220631662,0.7631285723976495,0.5810094453667398,-0.0007283056686023861,-0.2501476938564235,-0.19714189537788887,-0.5862759689433569,-0.8037662254874366,-0.931246336966271,-1.064626076208994,-1.116216109772345,-1.2652676087565013,-1.3818142662965556,-1.6819754911653904,-1.7577877605403465,-1.677516641294636,-1.7189054995861244,-1.7777833664596,-1.8173699882556087,-1.7539423466982345,-1.770219215198961,-1.6571235569214,-1.5012948462831375,-1.5881117927689457,-1.2961673501479927,-1.280969825775631,-1.0541567598309947,-0.8970953827761057,-0.7061951611895081,-0.6801359361395646,-0.5076138317537685,-0.4221632426271688,-0.29926935455300974,0.033989986881473366,0.13487583511631682,0.13264381286004276,0.19599490890963436,0.2612355276446854,0.2582792995545684,0.164134045427905,0.2578665245189765,0.20502755839373793,0.22328638330347023,0.24731640161983914,0.22475994868489474,0.017311590387035124,0.03042313434829769,-0.13299227246076437,-0.3951883296556337,-0.5374863406890046,-0.3553025019317205,-0.5012235609880297,-0.6539210345757802,-0.9550266078787379,-1.3076970421877543,-1.3222091622945518,-1.9774337153302601,-1.928772787631114,-1.966129721742591,-1.9796207786261015,-2.2952454855820044,-1.9784651383547798,-2.3376146929933896,-2.4681410447501704,-2.3244521700361567,-2.2255565110787296,-2.2437032735406923,-2.269296254493117,-2.206339847176686,-2.096994416745631,-1.9993659662635481,-2.0171366903147114,-1.940923228280143,-1.678790410423142,-1.6958906835025789,-1.4691042625316462,-1.256343287298316,-1.249149877712259,-1.1434652004441552,-1.0107233196628362,-0.8781687953628132,-0.7908366563159337,-0.6303016768033995,-0.5186972003729993,-0.4363697277582663,-0.35363631660545747,-0.2813381985340721,-0.187778028040093,-0.12472027176811241,-0.18493057684146513,-0.01087809201496637,-0.07366210773358654,-0.15858813599305108,-0.12339268872135484,-0.17484271738919047,-0.20437229278824254,-0.22490396912028887,-0.30588439619484875,-0.3981857451259213,-0.5271905585568312,-0.582896874825106,-0.6550235246478813,-0.8236745189008329,-1.1812963288788247,-1.2519499497165405,-1.3239041625052117,-1.2559531420514232,-1.3390732479254501,-1.4191802975109864,-1.5226716676205956,-1.5973924114459759,-1.9341756807593968,-1.6395673737164969,-1.792774401032003,-1.8256971709946044,-1.7988873722572336,-1.7795395105541303,-1.6755022002862843,-1.7468138811810954,-1.5897069283368892,-1.6042308356834012,-1.5472154681726655,-1.390010512311305,-1.2642932197972063,-1.089054249883235,-1.0482104524653078,-0.6782003331524623,-0.5847200630107504,-0.5442502025544302,-0.44493196266057,-0.3180598143488859,-0.2051045674595177,-0.14876417477718912,-0.12168601942854072,-0.023675483462684326,0.15305082443928197,0.060899376944999396,0.09310904321361743,0.07559487893590335,0.07657505351085964,0.1685892430757688,0.04574065536359145,0.19413116664313826,0.09714819626523886,0.2169614896522909,0.07457503803169852,-0.0240602795611809,-0.011390899767735725,-0.23495374998204097,-0.30259057060052297,-0.5590483287909596,-0.5435094785251693,-0.6398420406787191,-1.4106691476585491,-1.3920432823850089,-1.7284081669484714,-1.9008369022897291,-1.4114419532170421,-1.4336717720732186,-1.7194035304898456,-2.0260782417348504,-1.650435917786333,-1.3257871627573417,-2.0614588258365045,-1.927981065363798,-2.4310019971104047,-2.2494115322948596,-2.111800681937769,-2.0494231684626714,-1.9470197259395972,-1.8779597981294902,-1.8109370534504772,-0.92000466468351,-0.7196993552689582,-0.5611410980687932,-0.4259454163240402,-0.20400080149382033,-0.1818245410797353,-0.06817416191690524,-0.026546302028627577,-0.016337059521110604,0.07011182226823924,0.19900682967021063,0.2556343356743327,0.28120299921515357,0.4848099184725185,0.4077480307527397,0.47092024275205263,0.5664596635442514,0.7340172307621879,0.7285973071006259,0.7936309785784096,0.868982301721412,0.930382490379031,1.1103866243257838,0.9268642005051182,0.7723358197049379,0.6097351788738988,0.677756980552708,0.7249402551704607,0.4633121797470485,0.47949979001681264,0.45007660111437997,0.26653470308690574,-0.030069432505138542,-0.13687062779173156,-0.21327932871230942,-0.3115307268421804,-0.4049022552441389,-0.49952717188665163,-0.5610220979791989,-0.4930190368802899,-0.6775392768869252,-0.776967382885263,-0.7476355890049259,-0.7346771035638585,-0.5187658007582149,-0.650951125490147,-0.5215358931003852,-0.39800857368229087,-0.1820934037823405,-0.12358188862858774,-0.10476293174577256,0.11463076214276295,0.06083021992640823,0.2697082660484399,0.4617616968987426,0.6266867855925188,0.6837438230992468,0.7710903031881032,0.8456676484825321,1.0853184810499943,1.022163741423987,1.0674167825063738,1.4045739426026878,1.2851036562718579,1.551367488331634,1.5196954420277562,1.4286118144913798,1.6404927823795417,1.9116392264359863,2.04624474090206,2.0009044767846036,2.1630022850731536,1.6421387178192548,1.7633290653765619,1.3795294042399406,1.6368067250549596,1.476757728320864,1.1753520824431778,1.1996830746368332,1.09687144578162,0.8514089646968175,0.40970811077451075,0.16985366263010215,-0.08407610329120006,-0.4778173619253892,-0.7704371834162922,-0.8787472589598297,-1.0396921035633788,-1.1077342262726257,-1.2791073557846955,-1.1959054183647664,-1.306419896762917,-1.2273858007130984,-1.4185006562529636,-1.0980553473186814,-1.072690785005372,-1.1556141242034073,-1.0733741484825565,-0.9899766876078038,-0.9898304006526766,-0.624353514147728,-0.6071329326339523,-0.3675353816234751,-0.2052721643642161,-0.23928448789419396,0.21954445789796095,0.30940004805702936,0.39215539665673316,0.49224431392553736,0.6189735765501172,0.6884426019403092,0.5273920542607654,0.5884389979626097,0.6018424134091173,0.6653439768817332,0.49987427919880517,0.688612878848449,0.580752589197268,0.7123867859377042,0.8035559531886547,0.696448326091008,0.7213663373380458,0.972592474931533,0.8113431600943604,0.7001330079293534,0.5516512322471485,0.3555332967227754,0.21178018879457378,0.09314995738480665,0.10935896919687017,-0.13869014789766201,-0.328163317733961,-0.41041464160940144,-0.5166665673887906,-0.7547205174860195,-0.8026579530843815,-1.0593079421075373,-0.9690462881349916,-1.0417432174912542,-1.0247867096982302,-0.98540889260245,-0.9996348783820603,-0.9350592069997653,-0.9717639118994735,-0.9372422915710359,-0.7853747298491328,-0.8376109063344911,-0.6779596865857443,-0.7220662436480257,-0.5539146540004466,-0.46355266229914976,-0.25230914289528583,-0.28703936392965607,0.03809761124420584,-0.02557588168527205,0.07234329123471073,0.411710211691836,0.3920375248886879,0.37363393470417666,0.5817218808632745,0.5389001230339252,0.7145831729044642,0.6891657437266062,0.9380308748110233,0.9804177768368874,0.9338198963085169,1.0568559427266488,0.9541291431597776,1.1761296244766102,0.9250199399162277,1.064375214623606,0.8648529697964071,0.823842875034708,0.6944596524824914,0.5412561115364796,0.6690982600271779,0.4012061263106914,0.008825316014730435,0.11525751386490979,-0.16266344710042463,-0.45510567946941805,-0.4707675555738664,-0.45970087360695333,-0.5175480141682107,-0.8662882794479805,-0.7917615612492974,-0.9103779717635466,-1.196364936530604,-0.91341113106945,-1.059137531450506,-0.7978268055669226,-0.9938083972882729,-1.034192454733419,-1.0026233914958715,-0.973075992787634,-0.9283457343892723,-0.6114976569905196,-1.0291510750405004,-0.6670096229393965,-0.5117819528977589,-0.514709622888642,-0.4812083427113834,-0.14173375831145454,-0.3188964240582422,0.10384381507482339,0.28426778407620756,0.35801567971205034,0.6976037010857737,1.003613813695703,0.753959730832033,0.7642046507404545,1.1386025636184884,1.2351336645406665,1.2328463195029158,1.4105483404177228,1.4041794127195224,1.1216168687710046,1.6536161787921582,1.3715840584919785,1.3197053626640232,1.120579146605447,1.138267935328884,1.1611011979036037,0.8332047079334312,1.0312578082266926,0.8755654078226388,1.0635331931375038,0.7926288104798594,0.803028129288154,0.55200311056581,0.5937444854571823,0.3310222868770051,0.3957647206433999,0.3478050086234977,0.46991800410443235,0.16390230246228374,0.15516865923618045,0.3311997042212981,-0.05531481988472852,-0.28423539357667127,-0.24151550638216712,0.1496198360639386,-0.24107898718243895,-0.08216633662665691,0.021031039944525685,0.13326454828209663,0.022242583285100065,0.20093166084496422,0.3274744525350707,0.6597040336668355,0.5852698217327504,0.6518129743790336,0.9074890031988283,0.635520492057436,1.0162069839970125,0.9570528709916961,1.2260327340113188,1.308379575594614,1.5474574018467948,1.6745625267918944,1.1294165518201082,1.4803059842045787,1.2544297232342294,1.5841195661970842,1.6060710873499306,1.756338692091651,1.7524856679067604,1.3549700412166676,1.334483691900543,1.6025251518501287,1.2604492237773834,1.2651953495138064,1.0861250110191387,1.0747568566357157,1.0221758635825147,1.2167956194703493,0.6574010701723906,0.5873075880100395,0.5697259447923719,0.46204489394022,0.4356670624900476,0.19514562379060935,-0.01848448681119647,0.0052993412387372955,-0.195547706869124,-0.27686124068763374,-0.19778739028627573,-0.06994100284949108,-0.44962441495371586,-0.2088542929461939,-0.38194644489198504,-0.4992236984982375,-0.4005778090453842,-0.27368389402266396,-0.32379837961309355,-0.21838959498325497,-0.16315889242701245,-0.1018503725769365,0.1566025330414567,-0.0410287410663101,0.0284385118823533,0.29515733297241853,0.45745962608009416,0.698202612108122,0.7398720010129416,0.7150358635240671,1.1361181506231963,0.822515036060841,1.1774370110319399,1.3132917595148654,1.3430822970964404,1.3466689173290074,1.481287791270721,1.5921235556033804,1.6052524788313909,1.3191075370676737,1.083962258318488,1.6026751647802893,1.7887908964850252,1.3009665120475344,1.6789292497162036,1.9097134518911223,1.777107383448507,1.280552140175911,1.1555328535301739,1.295233541354544,0.9584405251732502,0.987877958158674,0.921643601093159,0.5190699991977243,0.3392705788242824,0.41777275012539533,0.4217667589778568,0.16557874241212472,-0.04168996458106715,0.3095336779540438,0.05330350986361865,-0.29767443804778765,-0.45104656772592316,-0.1730910767898761,-0.5201277266249168,-0.0848917574637796,0.030083744829602972,-0.1473280943216733,0.3219326977878024,0.6519239437283603,0.24353162910058404,0.29487120026354097,0.3857910714708348,0.6382455326583699,0.6935814226709943,1.2440713203280758,0.7125852483085768,0.9483670612344314,1.6780348376233087,1.3939581450818,1.6195515990148988,1.7649970672777693,1.6640038680219487,1.6159864429906414,1.819978838761105,1.3555897568876545,1.58818816692069,1.4604312489486935,1.7934586199830103,1.5786385976237514,2.0563245607629947,1.8873381698699228,1.757830979379784,1.6720470238168186,1.5869832281156648,1.2992981684064433,1.6701018114940642,0.9264061573765047,1.5277344694313084,0.9549260404656964,1.0853818889581148,0.7716819865074762,0.40322358915281525,0.3746092939129478,0.48406340170726947,0.35523375088842357,0.29251887529942594,-0.2980709877771592,-0.12742019943856212,-0.5712048249419303,-0.611397152896605,-0.3143424667554036,-0.5983363654698581,-0.6581229688687829,-0.7434233425661562,-0.9403489459306918,-0.9309837941075615,-0.895213196735891,-0.780611042029598,-0.5297291562717239,-0.513446412932303,-0.4893617154010557,-0.5004785416600112,-0.3101804189361486,-0.48790688261125165,0.056252498190742725,0.07506621577145521,0.3083490670104814,0.16597529473209297,0.26282569880800416,0.5075538528218553,0.40662126834145457,0.7513674836159938,1.0522478687951313,0.9720422700995275,1.0218568482395587,1.1563404086744673,0.9797418516978169,1.1123828464628032,1.2947242849311273,0.9944098977811647,1.0715086145576136,0.9091927501572206,1.015204642208696,0.8042813892957387,0.8770621760039237,0.7775718748786611,0.656941059093113,0.6983294479408652,0.401140918516399,0.44239393515305353,0.2384289614339271,0.28214053249307797,0.07378648901896381,0.10549408593798108,-0.12881587125349805,-0.21799768661632132,-0.4068986884444922,-0.17066417193282335,-0.46945852740137384,-0.6918732001494935,-0.8197569696571085,-0.5945610693685183,-1.1728804316763801,-0.6578792061071609,-0.9132853099068082,-0.7593922906883439,-0.5100914160950558,-0.642182050463756,-0.5911989054650035,-0.420144036919303,-0.4125860078863559,-0.1677502781471966,-0.248863971547468,-0.1523321471367094,-0.04661800253227478,0.10583652720845,0.07663620114278143,0.4206596039647033,0.5089803574454648,0.63064377599736,0.7857031071603265,0.6717818094672975,0.7461936010733258,0.8747115246783674,0.7322619448986113,0.8305463634394763,0.6511035405797221,0.7962164039050508,1.0403403853418438,0.7599802293742837,1.1320387105314382,0.9266908456464709,0.7209956041005245,0.6884753669127291,0.5176545252542225,0.3985678297299951,0.20297034802377828,0.05945379089729614,0.06810360172331587,-0.26456588901029965,-0.31428753086400457,-0.5207339243457204,-0.7622778110068574,-0.6479202938878633,-0.7728604283001972,-0.8639108356560362,-0.9329418875464626,-1.0464992383870382,-1.1025358195486825,-0.9727617660271936,-1.0234906706612348],"z":[-1.9572960504675927,-2.576085894776404,-2.1233411112422953,-2.2238295838800166,-2.242043074594586,-2.0536458508569444,-2.3181019017266324,-2.1033391786333664,-2.5547576285927485,-2.42429078568252,-2.3594922668257934,-2.0750952132392784,-2.0337997144076136,-2.093369273970468,-1.8064520670756774,-1.8384634645049545,-1.7282991297642214,-1.2929568144413581,-1.453075004625773,-1.1988640966541073,-1.614731805390651,-1.3874323682083012,-0.8269644131627322,-1.3184830573857484,-0.7900492922429696,-0.7220856053722895,-1.219791914941011,-0.48869129053159976,-0.9607954134197709,-1.1177062080720823,-1.6080873931755137,-1.4538751806799146,-0.45519442942004074,-0.7462475548526598,-1.352086152704078,-1.3738035507246027,-1.597427749142889,-1.0838799408969784,-1.0915115312248072,-0.7955213207506994,-1.093086297288844,-1.8772833848805703,-1.191242444608701,-1.819344976439525,-1.7723402048715748,-2.05014749475738,-2.307300254947021,-2.2527890747894443,-2.0805406144626235,-1.8366522845020175,-2.5464707103060555,-2.265914254456458,-2.4897443026035453,-2.33616086301173,-2.2287752705645714,-2.0578928015711924,-2.4045162806121803,-2.1794230254458196,-2.1587562033988545,-2.20421830618104,-2.0893621446736703,-2.0093475101569367,-2.020516076377732,-2.0410685578115455,-1.6011197691970664,-1.8053438982307979,-1.858912841712771,-1.8388788713098874,-1.5144862433503643,-1.442242831364663,-1.375769308898287,-1.1393011810296236,-1.493925428597235,-0.0771589158013665,-0.7721629352703806,-0.9808992582082644,-0.23446371314360684,-1.5626142288071252,-1.5869388598592173,-1.0791641434191563,-1.4847641525551343,-1.4985745509264845,-1.5390875280604355,-1.3359928546829862,-1.4007528235224633,-1.3699501272220014,-1.1175843814892323,-1.2551799222969346,-1.1521334245243475,-1.609031735701719,-1.3451986973127545,-0.6763928855520016,-1.1920168368047295,-1.3210364738436169,-1.4644069523193841,-1.8089299257653706,-1.7777267377560342,-1.4032994799394527,-1.6268691000649977,-2.081056053519406,-1.7900768672820573,-1.8215845183244728,-1.7163958145448968,-1.560266966895586,-2.0324690495512705,-1.9141893972970105,-1.5580573396073985,-1.5640452623095504,-1.8652486726990232,-1.8114573208622833,-1.6952387595821443,-1.922924522925572,-1.5339510668903675,-1.9522557972223311,-1.789120990892562,-0.8304567598854494,-1.5074334886522454,-1.4714048610120816,-0.4140349201812192,-1.2412769512984256,-0.7163838784621273,-0.41642191124472794,-0.2260101838032676,-0.7099838589058467,-0.41221756165589873,-0.5420704612307495,-0.18066955676388596,-1.6675262235975785,-1.4883036401322958,-1.617026750667252,0.02283902369712013,-0.19682395607327588,-0.33736295312875875,-0.328341236907405,-0.4555439519742411,1.785313342869938,1.7140065514128873,-0.26188560422626506,-0.26874003128561463,1.6875816952423242,-0.30266647400088875,-0.5406522144370249,1.5880632601857492,-0.2898456567330526,0.3140560844863213,0.6096234388781229,1.102229252114305,0.8023038460841514,0.6432274007479534,-1.029958274599526,0.9554215748984948,0.0736994197487831,0.3152969185188176,-0.029145679950143773,0.9096582062467757,-1.4601488292863258,-2.114088665748413,-1.5334468750223464,-1.4686739911086357,-1.8008908823352734,-1.5226941157528002,1.0556689189624624,1.1293562388875722,1.3717892286333955,1.3329074128887508,-0.7031460943273897,-0.6627318826761583,-0.7313733875056022,1.6145741062062977,-0.8550017349334856,0.02772962826087044,0.4596638218692264,-0.9879783836782502,0.27145438084112183,1.4638563404175242,0.38972473563537713,-0.49736003114582,0.4481423532614977,0.05650253914982219,0.2082111409058894,-0.054370465504178594,0.3723663223479012,0.1289192148165995,-0.09890580125397169,-0.1386583532479226,-0.2973582449046467,0.2262249299402016,0.1886215545377259,-0.09133296700337233,0.08676168746900263,0.13219871460909907,-0.1960908368804229,0.0988216750403427,0.17780413149619329,-0.03626444969442948,-0.34610934310520336,-0.46600606140490664,-0.3437727764806937,0.16464768459967746,-0.33339917826702575,-2.3277536313180587,0.120986441939199,0.3938886677264873,0.08948865993805553,-1.467406069030904,-1.9551691769985062,-1.5680909208599603,-1.9860815209506921,-1.8451145354481777,-1.6694738572875283,-2.076241378659632,-1.8869688648426322,-1.8448909730908598,-1.2242587616884786,-0.6951397624900073,-0.9424858477367963,-1.3013063395134168,3.1668312076443086,3.1325094735463423,3.211643612308493,3.2336803717074916,1.2379768522107617,0.015991242572523177,-0.028467398387814986,-0.48919765723251357,-0.023808048890161547,-0.2601382125931819,-0.37995941543640177,-0.5234047850568307,-0.4913458808977047,0.17873596291933594,-1.0442610285256375,-0.5710437250205556,-0.5203775698534662,-0.02008616689251735,-0.12453665100720938,0.1630844299143773,-0.5107238368028002,-0.34697461678217617,-0.2017014793678621,-0.1409518004264336,-0.11017674048478356,0.08282997467693946,-0.7775847372871388,-0.08667711744423864,-0.010159007421916444,-0.13532163128299782,-0.08686112298251022,-0.0167716448844044,-0.25207305108810013,-0.27683761052866634,0.024734701023688833,-0.9904708018032194,-0.42935820659563845,0.0690389337537359,0.03623611545959263,0.03480883539213436,0.20887567251056624,0.3256917941093684,0.34380920994901276,0.3195599652274022,-0.3874056023470224,-0.8302771774700126,-0.8213844060485067,-0.8723475214361808,0.7113475325621341,-0.7332718228539311,0.2257544092903119,-0.54049459333469,-0.09588248685046798,0.7104892398033928,0.6787233454014588,0.7467601854344608,0.3165683597734462,0.6964678530564747,0.5259146896102063,0.7633924500906475,0.3199764614733921,0.4373570122920341,0.14624621483443956,0.09399089379291133,-0.025481722899977945,-0.038267372565900425,-0.02372017467860573,-0.1238049917659866,0.036412972461353166,0.005175927792057514,-0.05154695683772313,-0.022708049660081612,-0.05086874538559527,-0.33882318113504234,-0.32456519784367555,0.3200115233305071,-0.14424618748171492,1.1361338272897892,0.33020674209290723,0.6957930835576497,0.4619773776386453,0.8119895891934018,0.7092533300227639,0.7946178086712467,0.7765777750902112,0.8742494176160854,0.8970504850066414,0.76227115532708,0.969530862276098,0.7836103691880794,0.9853217971608853,1.0086436289529792,0.987713248325124,1.0651630162620769,0.9681085195744462,1.0342156521262313,1.2965297339891293,1.2438978044055282,1.1700476285454091,1.2204736700585164,1.2242711686883763,1.4621898538154032,1.1779815790528123,1.3453722901019587,1.0009249457025995,0.9208934972422781,0.4544789486127794,0.38120357411955447,0.7680353014354904,0.7471598770573779,0.4773132825357267,0.6116747897947258,0.5764386257737806,0.7252269627529536,0.5968251277589883,0.38914887318694186,0.35834172818448956,0.10440178376076456,0.39125220717148723,0.2984118102042651,0.42460726408660504,0.2819483436192487,0.3007539254380912,-0.04641501917677207,0.1569914485028237,0.17240622069322964,0.2776492232117646,0.2639310046003418,0.37529797891902494,0.18447661218425357,0.21219348827049114,0.2450682761312807,0.36626137688245425,0.3409587496311393,0.3772151225015112,0.35766468810070956,0.5004075030011528,0.5737879905287865,0.42390138070312944,0.6383355424446207,0.7764358133087805,0.682134204496764,0.808705401434387,0.8614232374081768,0.8867258566925051,1.405378187503034,0.878865093844394,1.0877735913109505,1.1709080370429894,1.2417133198949781,1.3455385350654512,1.4988987751657346,1.512318277810087,1.608653787575403,1.4203227426795455,1.3837602884570495,1.401453157527038,1.609664011472415,1.793607887088436,1.2112922448186894,0.8430653701766115,1.2215132246127394,1.0321920252584056,0.7455418999398593,1.0368198984679322,0.7029991917077495,0.6513444007922173,0.7053517038561196,0.5754901205445035,0.7302182679360676,0.7140678836653985,0.7067289392507006,0.8540140668186931,0.7149961188143333,0.26478318341566076,0.6654768418420272,0.8229745706554986,0.4128960675596241,0.4165287562048445,0.6867555683512495,0.658201602258905,0.6072882481310258,0.5556850446706562,0.592993681416052,0.4452400080839597,0.4118919130298815,0.6601723441009699,0.5508749716185047,0.34680483104014787,0.2912530832008347,0.5554516324859969,0.6579684319390959,0.843955069611118,0.7847380771382887,0.9290221781169877,1.042368997399199,1.190413549132444,1.195399047617034,1.1081687839668461,1.224007522377791,1.0944303637612698,1.3838374299632974,1.3768491851490137,1.4177577992766506,1.3769479333221784,1.2822825543176444,1.3217880429472104,1.5509416660801592,1.3283869021356052,1.4244373836284951,1.0585290718285862,0.836131020836776,1.0808147988030241,0.8744070786725955,0.9772741596652473,0.9677506478903325,1.0082774519548385,0.6930804021080251,0.8116344880010203,0.768397911540554,0.8490977341827358,0.8955517349680823,0.6759004953620692,0.6131423948432643,0.5753558411454845,0.6895869110477101,0.43836629116992853,0.4203119987384982,0.4266545588505666,0.3301220030298455,0.2658718451580143,-0.0598254659813718,0.3247978627746754,0.31119529021079295,0.3370673701770752,0.10552531012393593,0.3472810636374821,0.46759667298130947,0.33387407329425256,0.33403471248133404,0.6142363121417074,0.8292805209301765,1.0557620995774484,1.0927899653905517,1.1046726612188906,1.0993264894943167,0.8787640351138903,1.3645378596239066,1.1479859836072035,1.0481053603940638,0.99269058690218,1.2419407177860113,1.1794088123495334,0.9342973561777349,1.0326226434452461,0.9795149909298061,0.9151164677303967,1.3211227801840735,0.9755183736253261,1.0372243332363573,0.817404215895691,0.8516394203120556,0.8261585690625046,0.7974458956248404,0.7262465240427917,0.7784376720345209,0.7591603132882149,1.0278190204297835,0.5976089627320523,0.5386795404024399,0.7152074664489465,0.4742797085126007,0.4419050863943965,0.23692740528905465,0.2474401896208068,0.20691691781112156,0.26169031189566433,0.30741431464131314,2.4668132588073894,-0.12904501611896954,-0.026269780297452982,-0.053732868853836886,0.09852584954636118,0.2660295329730164,0.07197754098499208,0.14279830595686402,0.24422127041774414,0.3802944817540256,0.5496903405698266,0.565151710191187,0.6639504781498831,0.6233399758369752,0.53010074169847,0.8003729557887376,0.8949937321120954,0.9154767359628259,1.0054423903027983,1.002887464339312,1.053440050001489,1.054298579196376,0.9948320773388196,0.9459091299537498,0.7084177048904596,0.7778979494445605,0.9363239197493985,0.8935131304963988,0.809042359660214,0.8607733350760132,0.47352561202484766,0.7417811157025651,0.47887622934573854,0.6260638847337171,0.6940038325629657,0.4713904034733247,0.47670580250160666,0.5052779474982746,0.3033351313957716,0.16346696247191744,0.37027526217124684,0.07401743670704329,0.29130595300925677,0.11550605123472425,0.11006147658558675,0.25595178603293334,0.11794743961419321,0.14171253291256164,0.033644696052598934,0.10286114424843451,0.11151428710315679,0.14655438323309647,0.030940603303081526,0.06670980657758104,0.058340588291827554,-0.39315354476684167,-1.0905187080744356,0.02581443405275287,-0.23645867620376665,-0.1820568387013636,-0.09731305038513743,-0.22193213712780818,0.17004741984537297,-1.4162806168255553,-0.7443521409473162,-0.43805384563801786,0.5044121181837264,-0.28319553670872566,0.9068583380442607,0.2148302068417476,-0.08248641674381704,0.5041343753427936,0.8362309662268281,0.9650643077842004,0.5898295643542504,0.7083983926304486,0.724290008151053,0.6784316624715766,0.7479293291975139,0.10014043308247393,0.5004729696093507,0.2925175889407804,0.4408822169289959,0.44656649441251756,0.3717277223935411,0.3703678453256756,0.394572638005181,0.3793582397827931,0.21497148345323727,0.32618866464487234,0.308193203098309,0.40117865018463367,0.1946223651524547,0.2534795703790004,0.42563674471118323,0.1321553910930148,0.20832114089241227,0.35485711644909923,0.22460249019570935,0.2818863601409257,0.2676640802172117,0.20961053114362801,0.37511647648138585,0.3932436606784942,0.3620608081628921,0.5429018098504973,0.47230984061298775,0.2715698237983285,0.6627463304081809,0.3315933352691905,-0.1311197896739587,-0.14306046068700673,-0.04356678009835947,0.5257585473117569,0.28799380300453264,0.3481881397228554,0.5523242776708542,0.3783628667679588,-0.3546693051108449,0.8100449511035522,0.7071536061623371,0.5872636664415484,0.7807286523295329,0.697718245459843,0.9385395661397092,0.5331909203438016,1.0097230776997734,0.706986919394347,0.5674431131570037,0.6646972743575864,0.7872791230598719,0.583195588052043,0.3679712157232703,0.5536145655894753,0.4472929345091629,0.7260012564613396,0.7236642550297334,0.3693848008091048,0.5654561615477975,0.5165661782079554,0.49521670129232115,0.6627021652386904,0.9919555349366721,0.5255239186485889,0.44675843865304166,0.4496172252657303,0.7454171800945664,0.4897010191929506,0.400192411922013,0.7904910402915378,0.4840197572272592,0.9281422477393553,0.25006253593814043,0.5747772848564796,0.7862370327058917,0.23314777253471322,0.9468123189403388,0.48720776939729576,0.5792610402530427,0.3852382398187477,-1.197810769099102,-1.30825145215544,-1.2051403666233838,-1.210269166170323,1.388014822911191,1.6796158959335152,1.3675095313227388,0.7318409447043043,2.1417629393574296,3.352913815728376,1.07320045103205,1.7728378772337607,-0.1581609163788842,0.3146835475229228,0.8149863457761024,0.7544226438590225,0.6766133773530857,0.7089373365108347,0.37174364020739614,2.98474814331023,3.1967865691118797,2.9877988214077895,2.913733086188079,2.7983531821297887,2.749817922691756,2.507001197611898,2.412321416558668,2.5146963072686117,2.5668027521056933,2.623545691289915,2.5590498214967448,2.3066287545575954,2.4174357936995166,2.3295562420983367,2.1610331437308568,2.0072374701872437,1.7935968172329617,1.6186139632819647,1.5427111020951576,1.4236545619730459,1.3691909286064083,1.381906905672336,1.4420583699799219,1.4793724378467454,1.3577733319502454,1.348667697080668,1.3449729123374716,1.3784612207969036,1.256070449581946,1.3209406780562314,1.230256391029924,1.3959714561394605,1.3604616504091886,1.63098386472816,1.5579947570048702,1.5930286625613352,1.6724848568241029,1.7490166216046001,1.6347577916316203,1.5342090878177979,1.7970283389043238,1.775801332355822,1.581594744298524,1.5771565882932983,1.7069604947338084,1.55932470864803,1.5181091770129627,1.4891728684257004,1.483774583896767,1.4215265381424327,1.4373039572666144,1.4129087616530627,1.5105951005934515,1.3673226064638195,1.3594812709198074,1.337495926137615,1.0037093300071456,1.4062782735156423,1.2350637835523226,1.0796078666892295,0.7086575767635993,1.018350967972579,0.5678727637975788,0.9482544232982544,0.6884200543696297,0.7458052044703752,0.6222739556458013,0.5599191422283665,0.39208913688859054,0.4458850759076672,0.36771311057740474,0.3455709032575022,0.41092352970791646,-0.05937868573080242,0.335463296464875,0.5100902786892878,0.15248865850110344,0.0804306620090351,0.059915444151187575,0.37820211907807166,0.2941584620864985,0.2969396490563533,0.2271893461766848,0.13005046107815196,0.36865036272197865,0.37671762027237643,0.6472327866581358,0.5829198989648953,0.5640701657667995,0.8172881524500216,0.6274703200435439,0.7581081958880276,0.4590248995486812,0.38434346284536136,0.7171466822276222,0.5162213083240382,0.5325369279055447,0.42004016944646466,0.3064602062631907,0.4292812408563895,0.5534339441529289,0.5358104037731612,0.1678826131659396,0.15942228111684015,0.04114453270849395,0.08777501059302371,0.03491942589567878,0.15783184080982904,0.0034864738905124636,0.2047300984764321,0.17523980647918194,0.27591246813400006,0.13998540657522243,0.35152800142907115,0.3717650710901038,0.20133909869354066,0.1854192011229966,0.1537446255922166,0.573345466019166,0.4827904734485017,0.5641541582722345,-0.13229629891003022,0.32083024315935815,0.5815317137864698,0.2153294850562404,0.6709175911782741,0.4126047826363026,0.167675512322817,0.9127745876433132,0.6943787585564022,0.13257238563185883,0.3719410421074928,0.6690392941929858,0.2683456154746842,0.5160882545690795,0.3394429010403508,0.6669585385923207,0.7203595348698074,0.8661123819956289,0.68717396753992,0.8783304768257293,0.8906481287702102,0.6273926273970469,0.5787922698766487,0.7549023513462607,0.6662401441255902,0.7604571755304789,0.3384721509493014,0.39205702479989685,0.8668413032468205,0.6632382525548585,0.4038632266533554,0.3075578575291044,0.4432988233632362,0.37378443352913043,0.15436016948384315,0.013078637685338537,0.30551983508217817,0.2825248783658513,0.02389006785769993,0.16486349116922858,-0.11508849652937057,-0.20274662438159238,-0.15814006471799485,0.0025314828075869464,0.34723844616131766,0.04612135727978686,0.15811284331695746,-0.003950184246317209,0.5480950522521093,0.3167912666719629,0.6409844770311081,0.713421950859252,0.594680032443968,0.842606180958238,0.3156360115914536,-0.17822046852016046,0.7987073773794123,-0.06040043570080296,0.03528009058242195,0.42900741824430827,-0.07434595121928068,-0.16288534196753962,0.10759276863768602,-0.03530840496359624,0.047329407879693135,0.032220550869477074,0.29637493978388924,-0.07116039246092831,0.40773162189955453,0.22094345791762954,0.10743494493075258,0.01963723558020312,0.21115221640353085,0.3286939268057887,0.019764898347057174,0.018250843056829266,0.018475180742364424,0.030895923295675636,-0.0010107258782116088,-0.1293628748622949,-0.14016437113586702,-0.24496720014191833,-0.07623583767493858,-0.1364063264413031,-0.17168363996561894,-0.2735010802576188,-0.25308611335615294,-0.5743047500265904,-0.120835991763528,-0.4277879858824208,-0.6493876281314255,-0.2590479075213687,-0.47944119455028783,-0.7382399683185985,-0.3807250618913624,-0.5812515370243522,-0.48145031271075256,-0.43620648837717557,-0.5345820530958639,-0.4185889049545951,-0.7063748679388758,-0.9800071231321792,-0.8447408800635235,-0.46429680508477944,-0.6969778358307412,-0.9562938149137585,-1.1589055401338617,-0.3298884493034506,-0.7351951187540435,-0.6755769643314072,-0.6442996806125466,-0.7587683848760718,-0.9248610290485563,-0.8248129529757489,-0.5851219015213873,-0.7429416157161152,-0.5273209320020795,-0.6397797569934323,-0.5659064601776153,-0.6847529634829295,-0.8330622462057835,-0.8113544583558161,-0.6556194511193681,-0.626199420326385,-0.6655478144077748,-0.6422491995714692,-0.5524719265329842,-0.8264083817926501,-0.9489634330498039,-0.7285024583979902,-0.9553231007955851,-0.7893704039215642,-0.8100333819378843,-1.0832130144289784,-1.000253966419421,-0.6627252051465117,-0.8360298556067904,-0.9577850104390502,-0.7067655102794449,-0.7070526218255562,-0.9093140332776515,-0.7206640547019197,-0.722088854663505,-0.862565569134788,-0.9923277303455874,-1.0344285423731783,-0.6956683578606723,-0.7291105191761311,-0.4485887019378577,-1.1299889283823255,-1.2152077954893432,-1.201217107306025,-0.7927541298817561,-0.7607521015512866,-1.0470795806491213,-1.1496504099466012,-0.7629310061537355,-0.7609786921568641,-0.7925128593137053,-0.2918844401997353,-0.6934349434511107,-0.5749630477371716,-0.8824383248451666,-0.7458765435976161,-0.7447040374799583,-0.703352811581272,-0.514131893104997,-0.4466495062700906,-0.46052945723847544,-0.893287679592044,-0.7691811711731321,-0.7275114790183256,-0.5584883355629309,-0.8150592199839113,-0.5877338286821222,-0.2578380312111711,-0.8905286906383444,-0.4496037223087636,-0.562899546644088,-0.6174806966934763,-0.8721185029162347,-0.9258626993414496,-0.7827177808217507,-1.0305813244734459,-1.181379473754023,-0.932801350202886,-1.0898674162888815,-1.2624612710381016,-1.1701073474595054,-1.2005046399294779,-0.13644785982164467,-0.7626326961764653,-0.11390370819260814,-0.8165078285237503,-1.0059609451502487,0.24725538726045979,0.39774205586566364,-0.9147539205237655,0.6962813525098855,0.6478212432234894,0.5003543287413771,-0.866470169317146,-1.213859268130152,-0.9198916307800367,-0.9345206258366456,-0.9126790886548535,-0.8623754599892749,-0.9379651647129672,-0.6710240860323778,-0.6787370854545708,-0.486545092524302,-1.0015509166839458,-0.6357632930339855,-0.6665924557147581,-0.1113024940696044,0.357697418098047,-0.581474524453907,-0.24237945608816738,-0.5665510539878739,-0.6037050732341522,-0.6968897713154302,-0.5608175300416401,-0.7082103278441694,-0.6465431419904473,-0.6071520517183561,-0.5242685186093615,-0.5336136182398733,-0.7817037526959153,-0.5137637448695483,-1.182947313603748,-0.7221945660338105,-0.576073172588706,-1.0786670442938588,-0.585853378180294,-1.0118122896318338,-1.0568926522579383,-1.0565741415260053,-1.048343583107045,-1.0663280037639564,-0.8763461209990954,-0.15842150362945998,-0.7049085939904725,-0.9139439422535544,-1.3197266531043896,-1.1774573939219017,-0.6740565168452023,-1.075929939660521,-1.4003886876303573,-1.0294994374514808,-1.4773469497933993,-0.7314281837919064,-1.4186909944030326,-0.5655345709662271,-0.7310815887864107,-0.3958197314405532,-0.660551683802377,-1.5556264148909797,0.3088029463081105,0.08237108123007114,0.7606147048592183,0.31733931949237515,-0.4905014306490533,0.07698844851613523,-0.8978113678451087,-0.28052628835717225,0.8817048755013963,0.14219713625276328,-0.13156895856233347,0.17925318220782316,-0.5085153013855841,-0.5272623662452978,-0.660654869995545,-0.3555855054893255,-0.41829424966136824,-0.5286074070581196,-0.6713986537249135,-0.7679748396173918,-0.7145534227341258,-1.0543370166307597,-0.6346162251017003,-1.2432159046151467,0.10695922847720002,-1.012977312216076,-1.0439587745754222,-1.2975844486308818,-1.0281706549848744,-1.1943257398192793,-0.9851793046440758,-1.1293529983822772,-1.075608325309604,-0.8385872759541111,-0.8763053051657804,-0.7906155930862353,-0.8894775887531016,-1.0307900089145556,-0.7089072652787649,-0.8322885883646438,-0.7579648690155903,-0.8226359653519992,-1.3003493083805866,-0.7723976407924998,-0.8907338727501577,-1.0967631999526215,-1.0371667846183343,-0.7960560583939246,-0.9639679287142225,-0.6811380771444678,-0.9964072782161827,-0.4712338651426848,-0.7406386065002293,-0.6016128006123646,-0.7948525296072351,-0.5762009362841457,-0.112952913625494,-0.5808251236709394,-0.36565612202052555,-0.5305240896120553,-0.805997280758536,-0.40610258839904445,-0.4389336638725841,-0.48463717502247927,-0.32943597313233747,-0.5298614852724879,-0.22122188144425278,-0.2525643356910377,-0.6623932483971591,-0.26550730845010523,-0.5356436450599253,-0.409663517723785,-0.5745493984495056,0.01711336480399875,-0.743308293019634,-0.26806321330152694,-0.6505071120438417,-0.5403073992431362,-0.526520738547377,-0.5814090655941843,-0.5161089574047573,-0.5324743084454493,-0.30352572891951196,-0.9126550252058377,-0.556095189259578,-0.16951178741617792,-0.37026974411153857,-0.5954844289779667,-0.5788416160021447,-0.9621970817475286,-0.6676488129344196,-0.5036172970251802,-0.9453572965541578,-0.9475163724986303,-0.5266687584681087,-0.700455680230061,-0.17744589933657343,-0.5625301342229561,-0.405286436647756,-0.3606997477607355,-0.37746520737107775,-0.40584928718799573,-0.28449958051155816,-0.276282098988216,-0.3891716368011674,-0.28378565399190114,-0.28812178417065687,-0.1617160371341851,-0.1485152159982843],"type":"scatter3d"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"scene":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"xaxis":{"title":{"text":"PC1"}},"yaxis":{"title":{"text":"PC2"}},"zaxis":{"title":{"text":"PC3"}}},"coloraxis":{"colorbar":{"title":{"text":"Cluster"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"tracegroupgap":0},"margin":{"t":60},"height":700,"width":1000},                        {"responsive": true}                    ).then(function(){
                            
var gd = document.getElementById('970e7d10-ad81-4b7d-8c42-28aa141cd4a7');
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

                        })                };                            </script>        </div>
</body>
</html>
```
:::
:::

::: {.cell .markdown id="t4a3M3ng9J4t"}
Check 2 and 10 clusters.
:::

::: {.cell .code colab="{\"height\":378,\"base_uri\":\"https://localhost:8080/\"}" id="Pnr6-FYm9_m_" outputId="fc04e3e3-e7f0-476f-8711-48d452e2ff30"}
``` {.python}
# visualize the adcp data
fig, axs = plt.subplots(2, figsize=(30, 6))

axs[0].scatter(x=samples, y=adcpdata_scaled.iloc[:,0], c=labels[0], cmap='plasma')
axs[0].set_ylabel('Water depth')
axs[0].set_ylim([-2, 2])

axs[1].scatter(x=samples, y=adcpdata_scaled.iloc[:,0], c=colormap, cmap='plasma')
axs[1].set_ylabel('Water depth')
axs[1].set_ylim([-2, 2])

plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/10059b022e4166687fd04487744b2bd1226c7c6c.png)
:::
:::

::: {.cell .markdown id="784P-Y4THPtq"}
Detailed view on each of 9 clusters.
:::

::: {.cell .code id="K-t7qZpXIjxO"}
``` {.python}
# new binary colormaps for every cluster
clustermaps = []

for i in range(clusters_pca):
  tempmap = list(colormap)
  for j in samples:
    if tempmap[j]!=i+1:
      tempmap[j]=0
  clustermaps.append(tempmap)
```
:::

::: {.cell .code colab="{\"height\":759,\"base_uri\":\"https://localhost:8080/\"}" id="fj6LAFunuCfO" outputId="14cefa51-4ce4-40b4-c8f8-8589f94449f1"}
``` {.python}
# visualize the adcp data
cluster_plot = 1

fig, axs = plt.subplots(5, figsize=(25, 13))

axs[0].scatter(x=samples, y=adcpdata_scaled.iloc[:,0], c=clustermaps[cluster_plot-1], cmap='Reds', alpha=0.75)
axs[0].set_ylabel('Water depth')
axs[0].set_ylim([-2, 2])

axs[1].scatter(x=samples, y=adcpdata_scaled.iloc[:,1], c=clustermaps[cluster_plot-1], cmap='Reds', alpha=0.75)
axs[1].set_ylabel('Wave height')
axs[1].set_ylim([-2, 6])

axs[2].scatter(x=samples, y=adcpdata_scaled.iloc[:,2], c=clustermaps[cluster_plot-1], cmap='Reds', alpha=0.75)
axs[2].set_ylabel('Wave period')
axs[2].set_ylim([-4, 4])

axs[3].scatter(x=samples, y=adcpdata_scaled.iloc[:,3], c=clustermaps[cluster_plot-1], cmap='Reds', alpha=0.75)
axs[3].set_ylabel('West energy')
axs[3].set_ylim([-2, 8])

axs[4].scatter(x=samples, y=adcpdata_scaled.iloc[:,4], c=clustermaps[cluster_plot-1], cmap='Reds', alpha=0.75)
axs[4].set_ylabel('East energy')
axs[4].set_ylim([-2, 8])

plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/4b14b1d521eee88a51271e4783e47f373f139a69.png)
:::
:::

::: {.cell .code colab="{\"height\":394,\"base_uri\":\"https://localhost:8080/\"}" id="6AqlggYziYb2" outputId="c0726550-67a3-4d14-c071-5737b11fdf12"}
``` {.python}
# add the cluster column and calculate mean over each cluster
adcpdata_nodate['Cluster'] = colormap
adcpdata['Cluster'] = colormap
adcpdata_typical = adcpdata_nodate.groupby('Cluster')[['WaterDepth', 'WaveHeight', 'WavePeriod', 'WestEnergy', 'EastEnergy']].mean()

display(adcpdata_typical)
```

::: {.output .display_data}
```{=html}
  <div id="df-e6dc2ef7-76f9-47d5-8502-cc5a5bbf57b3">
    <div class="colab-df-container">
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
      <th>WaterDepth</th>
      <th>WaveHeight</th>
      <th>WavePeriod</th>
      <th>WestEnergy</th>
      <th>EastEnergy</th>
    </tr>
    <tr>
      <th>Cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>9.177391</td>
      <td>0.404261</td>
      <td>5.862609</td>
      <td>0.137741</td>
      <td>0.125368</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.397453</td>
      <td>0.280820</td>
      <td>6.777930</td>
      <td>0.449922</td>
      <td>0.449924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.984786</td>
      <td>0.149517</td>
      <td>7.101931</td>
      <td>0.282552</td>
      <td>0.298678</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.896707</td>
      <td>0.204837</td>
      <td>10.018261</td>
      <td>0.212131</td>
      <td>0.213932</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.750100</td>
      <td>0.205500</td>
      <td>6.410000</td>
      <td>1.649041</td>
      <td>1.832343</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.504707</td>
      <td>0.271933</td>
      <td>6.505200</td>
      <td>0.880137</td>
      <td>0.909515</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.116917</td>
      <td>0.225333</td>
      <td>11.021250</td>
      <td>0.599850</td>
      <td>0.596112</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.307373</td>
      <td>0.262696</td>
      <td>4.943814</td>
      <td>0.142420</td>
      <td>0.139793</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8.702251</td>
      <td>0.403288</td>
      <td>8.821027</td>
      <td>0.158685</td>
      <td>0.149175</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.850639</td>
      <td>0.289838</td>
      <td>7.862245</td>
      <td>0.112513</td>
      <td>0.123167</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e6dc2ef7-76f9-47d5-8502-cc5a5bbf57b3')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e6dc2ef7-76f9-47d5-8502-cc5a5bbf57b3 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e6dc2ef7-76f9-47d5-8502-cc5a5bbf57b3');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"height\":363,\"base_uri\":\"https://localhost:8080/\"}" id="IOBlyoIbr_JO" outputId="e315774d-30a1-4f59-d7e2-c43b353d8773"}
``` {.python}
indices = []

for j in range(clusters_pca):
  dist = kmeans[clusters_pca-2].transform((adcpdata_pca.iloc[: , :-1]).to_numpy())[:, j]
  indices.append(min(range(len(dist)), key=dist.__getitem__))

display(adcpdata.iloc[indices])
```

::: {.output .display_data}
```{=html}
  <div id="df-fa3567a0-9927-4b96-96bb-c69987d701a6">
    <div class="colab-df-container">
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
      <th>Date</th>
      <th>WaterDepth</th>
      <th>WaveHeight</th>
      <th>WavePeriod</th>
      <th>WestEnergy</th>
      <th>EastEnergy</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>439</th>
      <td>2019-05-21 03:59:57.805</td>
      <td>9.549</td>
      <td>0.365</td>
      <td>5.820</td>
      <td>0.022162</td>
      <td>0.163124</td>
      <td>1</td>
    </tr>
    <tr>
      <th>546</th>
      <td>2019-05-22 06:44:57.270</td>
      <td>9.387</td>
      <td>0.130</td>
      <td>7.365</td>
      <td>0.312589</td>
      <td>0.471319</td>
      <td>2</td>
    </tr>
    <tr>
      <th>265</th>
      <td>2019-05-19 08:29:58.675</td>
      <td>6.573</td>
      <td>0.190</td>
      <td>7.000</td>
      <td>0.298709</td>
      <td>0.336520</td>
      <td>3</td>
    </tr>
    <tr>
      <th>215</th>
      <td>2019-05-18 19:59:58.925</td>
      <td>6.787</td>
      <td>0.205</td>
      <td>10.060</td>
      <td>0.172660</td>
      <td>0.179855</td>
      <td>4</td>
    </tr>
    <tr>
      <th>455</th>
      <td>2019-05-21 07:59:57.725</td>
      <td>8.142</td>
      <td>0.235</td>
      <td>6.075</td>
      <td>2.010858</td>
      <td>1.440629</td>
      <td>5</td>
    </tr>
    <tr>
      <th>202</th>
      <td>2019-05-18 16:44:58.990</td>
      <td>9.148</td>
      <td>0.220</td>
      <td>6.350</td>
      <td>0.885962</td>
      <td>0.861600</td>
      <td>6</td>
    </tr>
    <tr>
      <th>207</th>
      <td>2019-05-18 17:59:58.965</td>
      <td>8.256</td>
      <td>0.220</td>
      <td>11.225</td>
      <td>0.608330</td>
      <td>0.592228</td>
      <td>7</td>
    </tr>
    <tr>
      <th>710</th>
      <td>2019-05-23 23:44:56.450</td>
      <td>7.183</td>
      <td>0.275</td>
      <td>5.055</td>
      <td>0.160146</td>
      <td>0.061208</td>
      <td>8</td>
    </tr>
    <tr>
      <th>926</th>
      <td>2019-05-26 05:44:55.370</td>
      <td>8.011</td>
      <td>0.510</td>
      <td>8.775</td>
      <td>0.225888</td>
      <td>0.147954</td>
      <td>9</td>
    </tr>
    <tr>
      <th>876</th>
      <td>2019-05-25 17:14:55.620</td>
      <td>7.667</td>
      <td>0.335</td>
      <td>7.925</td>
      <td>0.138892</td>
      <td>0.185848</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fa3567a0-9927-4b96-96bb-c69987d701a6')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fa3567a0-9927-4b96-96bb-c69987d701a6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fa3567a0-9927-4b96-96bb-c69987d701a6');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code id="IYDQHhzsURej"}
``` {.python}
from sklearn.neighbors import LocalOutlierFactor

neighbors = range(1,181)
factors = []
outliers = []
total_factor = []
total_outliers = []

for i in neighbors:
  clf = LocalOutlierFactor(n_neighbors=i)
  outliers.append(clf.fit_predict(adcpdata_scaled))
  factors.append(clf.negative_outlier_factor_)
  total_outliers.append(list(Counter(outliers[i-1]).values())[1])
  total_factor.append(sum(factors[i-1]))
```
:::

::: {.cell .code colab="{\"height\":283,\"base_uri\":\"https://localhost:8080/\"}" id="zaZk8k5qxSve" outputId="da1d7f98-6aa0-49b4-c905-f24ed9481a07"}
``` {.python}
# plot ololo
plt.plot(neighbors, total_factor, '-')
plt.xlabel('Number of neighbor points')
plt.ylabel('Total factor')
plt.ylim([-1300, -1200])
plt.grid()
plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/8195634c10e4e7c51683a374a68653ea18addfe1.png)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="vEGzYDgEldoS" outputId="bdb80f75-f909-4e83-ac6c-e89506cd9899"}
``` {.python}
# index of the max value
neighbors_lof = max(range(len(total_factor)), key=total_factor.__getitem__)
print(neighbors_lof)

# number of ouliers
print(total_outliers[neighbors_lof-1])
```

::: {.output .stream .stdout}
    42
    13
:::
:::

::: {.cell .code colab="{\"height\":283,\"base_uri\":\"https://localhost:8080/\"}" id="RqJZGZ8gXnAF" outputId="69f4868f-3711-4dee-a610-7be658ddbb55"}
``` {.python}
# plot ololo
plt.plot(neighbors, total_outliers, '-')
plt.xlabel('Number of neighbor points')
plt.ylabel('Total outliers')
plt.ylim([8, 20])
plt.grid()
plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/fabff27baa3bc6ee1d2c5630f5a09c39cede7213.png)
:::
:::

::: {.cell .code colab="{\"height\":244,\"base_uri\":\"https://localhost:8080/\"}" id="GgWnVJwZWhEt" outputId="917e1c09-ecb3-435d-e44b-a922e30643a3"}
``` {.python}
# factors
plt.figure(figsize=(30, 3))
plt.scatter(x=samples, y=adcpdata_scaled.WestEnergy, c=factors[neighbors_lof-1], cmap='gray')
plt.ylabel('West Energy')
plt.title("Factor")
```

::: {.output .execute_result execution_count="82"}
    Text(0.5, 1.0, 'Factor')
:::

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/854c992bfb2dfaab244e148d80513f89e60969b9.png)
:::
:::

::: {.cell .code colab="{\"height\":759,\"base_uri\":\"https://localhost:8080/\"}" id="V0LbXL8Lnloy" outputId="3103d10f-da2d-4668-cac1-18b7f239a5fc"}
``` {.python}
# visualize the adcp data
fig, axs = plt.subplots(5, figsize=(25, 13))

axs[0].scatter(x=samples, y=adcpdata_scaled.iloc[:,0], c=outliers[neighbors_lof-1], cmap='copper', alpha=0.75)
axs[0].set_ylabel('Water depth')
axs[0].set_ylim([-2, 2])

axs[1].scatter(x=samples, y=adcpdata_scaled.iloc[:,1], c=outliers[neighbors_lof-1], cmap='copper', alpha=0.75)
axs[1].set_ylabel('Wave height')
axs[1].set_ylim([-2, 6])

axs[2].scatter(x=samples, y=adcpdata_scaled.iloc[:,2], c=outliers[neighbors_lof-1], cmap='copper', alpha=0.75)
axs[2].set_ylabel('Wave period')
axs[2].set_ylim([-4, 4])

axs[3].scatter(x=samples, y=adcpdata_scaled.iloc[:,3], c=outliers[neighbors_lof-1], cmap='copper', alpha=0.75)
axs[3].set_ylabel('West energy')
axs[3].set_ylim([-2, 8])

axs[4].scatter(x=samples, y=adcpdata_scaled.iloc[:,4], c=outliers[neighbors_lof-1], cmap='copper', alpha=0.75)
axs[4].set_ylabel('East energy')
axs[4].set_ylim([-2, 8])

plt.show()
```

::: {.output .display_data}
![](vertopal_fe2dffe7741847bfb90c93d5a162f46d/c2cea51ff5e5a5be78527ee1f3b9946332c2c1ab.png)
:::
:::

::: {.cell .code colab="{\"height\":457,\"base_uri\":\"https://localhost:8080/\"}" id="KZyZ1Lhno40F" outputId="5db1b452-79cb-44b5-a17b-ea0bd9375330"}
``` {.python}
# final table
indeces_lof = np.where(outliers[neighbors_lof-1] == -1)
display(adcpdata.iloc[indeces_lof])
```

::: {.output .display_data}
```{=html}
  <div id="df-8cffada8-926a-4457-9a82-5a6a90cd077d">
    <div class="colab-df-container">
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
      <th>Date</th>
      <th>WaterDepth</th>
      <th>WaveHeight</th>
      <th>WavePeriod</th>
      <th>WestEnergy</th>
      <th>EastEnergy</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>217</th>
      <td>2019-05-18 20:29:58.915</td>
      <td>6.590</td>
      <td>0.265</td>
      <td>2.3200</td>
      <td>0.138710</td>
      <td>0.153322</td>
      <td>8</td>
    </tr>
    <tr>
      <th>218</th>
      <td>2019-05-18 20:44:58.910</td>
      <td>6.519</td>
      <td>0.270</td>
      <td>2.4125</td>
      <td>0.119482</td>
      <td>0.097392</td>
      <td>8</td>
    </tr>
    <tr>
      <th>219</th>
      <td>2019-05-18 20:59:58.905</td>
      <td>6.496</td>
      <td>0.245</td>
      <td>2.3800</td>
      <td>0.062736</td>
      <td>0.071739</td>
      <td>8</td>
    </tr>
    <tr>
      <th>220</th>
      <td>2019-05-18 21:14:58.900</td>
      <td>6.483</td>
      <td>0.235</td>
      <td>2.3700</td>
      <td>0.077718</td>
      <td>0.061038</td>
      <td>8</td>
    </tr>
    <tr>
      <th>455</th>
      <td>2019-05-21 07:59:57.725</td>
      <td>8.142</td>
      <td>0.235</td>
      <td>6.0750</td>
      <td>2.010858</td>
      <td>1.440629</td>
      <td>5</td>
    </tr>
    <tr>
      <th>495</th>
      <td>2019-05-21 17:59:57.525</td>
      <td>9.273</td>
      <td>0.270</td>
      <td>2.0700</td>
      <td>1.247442</td>
      <td>1.174273</td>
      <td>6</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2019-05-21 18:14:57.520</td>
      <td>9.225</td>
      <td>0.205</td>
      <td>6.5150</td>
      <td>2.072987</td>
      <td>2.065725</td>
      <td>5</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2019-05-21 18:29:57.515</td>
      <td>9.174</td>
      <td>0.205</td>
      <td>6.4150</td>
      <td>2.046979</td>
      <td>1.823695</td>
      <td>5</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2019-05-21 18:44:57.510</td>
      <td>9.084</td>
      <td>0.190</td>
      <td>6.5150</td>
      <td>1.768551</td>
      <td>2.355101</td>
      <td>5</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2019-05-21 18:59:57.505</td>
      <td>8.996</td>
      <td>0.195</td>
      <td>6.6050</td>
      <td>1.126666</td>
      <td>2.026475</td>
      <td>5</td>
    </tr>
    <tr>
      <th>500</th>
      <td>2019-05-21 19:14:57.500</td>
      <td>8.888</td>
      <td>0.230</td>
      <td>6.1150</td>
      <td>1.443117</td>
      <td>1.906792</td>
      <td>5</td>
    </tr>
    <tr>
      <th>503</th>
      <td>2019-05-21 19:59:57.485</td>
      <td>8.424</td>
      <td>0.175</td>
      <td>6.3000</td>
      <td>1.557304</td>
      <td>2.323240</td>
      <td>5</td>
    </tr>
    <tr>
      <th>664</th>
      <td>2019-05-23 12:14:56.680</td>
      <td>6.619</td>
      <td>0.095</td>
      <td>2.4550</td>
      <td>0.217628</td>
      <td>0.146865</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8cffada8-926a-4457-9a82-5a6a90cd077d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8cffada8-926a-4457-9a82-5a6a90cd077d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8cffada8-926a-4457-9a82-5a6a90cd077d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="r9eRwp4svrxo" outputId="60322922-6c7b-4396-f202-2c198c68c9a7"}
``` {.python}
print(np.intersect1d(indices, indeces_lof))
```

::: {.output .stream .stdout}
    [455]
:::
:::
