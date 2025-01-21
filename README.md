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

-[Yearly scale EDA](GEMS_TCO_EDA/2years_mean_var_sem.ipynb): Presented time series of means, variances and semivariograms per hour from January 2023 to December 2024. The plots show not only ```short-term cycles``` but also ```long-term cycles```.

-[Monthly scale EDA ](GEMS_TCO_EDA/var_sem_timeseries_foronemonth.ipynb): We present time series of semivariograms, and variances. It shows that the process is ```anisotropic``` and this needs to be reflected in the modeling.

-[Hourly scale EDA ](GEMS_TCO_EDA/sem_crossvario_houlryeda.ipynb): We explored data on an hourly scale. ```The cyclic pattern``` shown in the ```semivariograms``` indicates that we should fit the data with a model that can explain this cyclic pattern. Secondly, ```asymmetric cross-variograms``` on some days imply that there is ```space and time interaction```, hence we should consider a non-separable model. Lastly, ```latitude-sliced``` data shows ```spatial non-stationarity```. I plan to ```detrend for 5x10 spatial points``` in the N5N10 E110E120 region. 

-[Ozone distribution analysis](GEMS_TCO_EDA/TCO_VS_Staratoshere.ipynb): I compared total column ozone with stratospheric ozone and analyzed the time series of means and variances of ozone values over time to confirm that most of the ozone is located in the stratosphere. 

## Job orders for Amarel Rutgers Cluster

-[Relevant Gaussian Process related exercises can be found here. ](https://github.com/cl20813/Gaussian_Process_Exercises)        

[Compare full likelihood vs Vecchia approximation, simple spatial model ](Exercises/full_vs_vecchia)                
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




