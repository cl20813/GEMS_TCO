# GEMS Total Column Ozone Project (Package)

## Setup and Shortcuts
-[Create and share environment on a local computer](https://github.com/cl20813/Softwares_Setup/blob/main/install_python.md)        

-[Create an environment on Rutgers Amarel](https://github.com/cl20813/Softwares_Setup/blob/main/amarel_environment.md)
            
-[Install ```gems_tco``` package on Amarel](https://github.com/cl20813/Softwares_Setup/blob/main/install_mypackage_amarel.md)      

-[Copy the dataset to Amarel HPC](copy_gemsdata_to_amarel)

-[Amrel_guide](Amrel_guide)

-[Domain knowledge](references_domain_knowledge)

-[Link to OnDemand HPC](http://ondemand.hpc.rutgers.edu )    

## Research Proposal and Exploratory Data Analysis
-[Research Proposal](GEMS_TCO_EDA/Spatio_temporal_modeling.pdf): The main goal of the project is to develop tools that can help in modeling the spatio-temporal Ozone process.

-[Yearly scale EDA](GEMS_TCO_EDA/yearly_eda.ipynb): Presented time series of means, variances and semivariograms per hour from January 2023 to December 2024. The plots show not only ```short-term cycles``` but also ```long-term cycles```.

-[Monthly scale EDA ](GEMS_TCO_EDA/monthly_eda.ipynb): We present time series of semivariograms, and variances. It shows that the process is ```anisotropic``` and this needs to be reflected in the modeling.

-[Hourly scale EDA ](GEMS_TCO_EDA/hourly_eda.ipynb): We explored data on an hourly scale. ```The cyclic pattern``` shown in the ```semivariograms``` indicates that we should fit the data with a model that can explain this cyclic pattern. Secondly, ```asymmetric cross-variograms``` on some days imply that there is ```space and time interaction```, hence we should consider a non-separable model. Lastly, ```latitude-sliced``` data shows ```spatial non-stationarity```. I plan to ```detrend for 5x10 spatial points``` in the N5N10 E110E120 region. 

-[Ozone distribution analysis](GEMS_TCO_EDA/TCO_VS_Staratoshere.ipynb): I compared ```total column ozone``` with ```stratospheric ozone``` and analyzed the time series of means and variances of ozone values over time to confirm that most of the ozone is located in the stratosphere. 

## Models
-[Full Likelihood V.S. Vecchia Approximation](Exercises/likelihood_exercise/results.md):

-[CNN integrated with LSTM](models/fit_deep_learning.ipynb): Still working on it. It seems CNN does not capture high resolution spatial information. Maybe I should replace it with Gaussian Process followed by LSTM. 

-[SLURM job order to HPC](models/deep_learning_cnn_lstm_slurm.md)




# Reference
## Job orders for Amarel Rutgers Cluster

-[Relevant Gaussian Process related exercises can be found here. ](https://github.com/cl20813/Gaussian_Process_Exercises)        

[Compare full negative log likelihood vs Vecchia approximation, pure spatial model ](Exercises/likelihood_exercise/slurm_full_vs_vecchia_space):
For this experiment, I choose data in July 2024, and used Matern model with fixed parameters: sigma 60, lon_range 8.25, lat_range 8.25, smooth 0.55, nugget 5. 
When conditioned on 15 nearest neighbors after maxmin ordering, result was as below:
```20x40 gird: (1936.3855, 1936.4033)```, ```25x50 grid: (2830.9476,2831.7986)```, ```34x67 grid: (5155.1572,5155.5592)```, ```50x100 grid: (10637.5962,10637.4082)```.

We can conclude that, for ```purely spatial model```, we can approximate likelihood by using ```Vecchia approximation with maxmin odering and 15 conditioning number```.   
Note that if the parameters are ```outside the parameter space```, then the vecchia fails to approximate the likelihood. Also it is important to ```approximate the likelihood even after changing the parameters.``` 


[Compare full likelihood vs Vecchia approximation, spatio-temporal model ](Exercises/full_vs_vecchia_spatio_temporal)  
[Compare full likelihood vs Vecchia approximation, gneiting model ](Exercises/full_vecc_gneiting)  


[Fit nugget parameter in a simple Matern Model using full averaged data](fit_nugget)           
[Fitting nugget parameter in matern model using Vecchia approximation](Exercises/fitting_nugget_vecchia) 

[Fit scale, range, smooth in matern model using Vecchia](Exercises/fit_matern)

[Fit non separable spatio-temporal model](Exercises/fit_spatio_temporal_11_1)        
[Fit non separable gneiting model](Exercises/fit_gneiting)                  

[Fit non separable spatio-temporal Nov.16.2024](Exercises/fit_st_11_14) 

[testing_algorithm_11_22](Exercises/testing_alg) 

## Debugging errors 

-[Common errors](errors.md) 




