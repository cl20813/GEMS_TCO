# GEMS Total Column Ozone Project (Package)

## Setup and Shortcuts

-[Create an environment and Compile ```maxmin.cpp``` file on local, and AMAREL](faiss_env.md)      

-[Copy the dataset to Amarel HPC](copy_gemsdata_to_amarel)

-[Amrel_guide](Amrel_guide)

-[Domain knowledge](references_domain_knowledge)

-[Link to OnDemand HPC](http://ondemand.hpc.rutgers.edu )    

## Research Proposal and Exploratory Data Analysis on GEMS Ozone Data
-[Research Proposal](GEMS_TCO_EDA/Spatio_temporal_modeling.pdf): The main goal of the project is to develop tools that can help in modeling the spatio-temporal Ozone process.

-[Yearly scale EDA](GEMS_TCO_EDA/will_use/yearly_eda.ipynb): Presented time series of means, variances and semivariograms per hour from January 2023 to December 2024. The plots show not only ```short-term cycles``` but also ```long-term cycles```.

-[Monthly scale EDA ](GEMS_TCO_EDA/will_use/monthly_eda.ipynb): We present time series of semivariograms, and variances. It shows that the process is ```anisotropic``` and this needs to be reflected in the modeling.

-[Hourly scale EDA ](GEMS_TCO_EDA/will_use/hourly_eda.ipynb): We explored data on an hourly scale. ```The cyclic pattern``` shown in the ```semivariograms``` indicates that we should fit the data with a model that can explain this cyclic pattern. Secondly, ```asymmetric cross-variograms``` on some days imply that there is ```space and time interaction```, hence we should consider a non-separable model. Lastly, ```latitude-sliced``` data shows ```spatial non-stationarity```. I plan to ```detrend for 5x10 spatial points``` in the N5N10 E110E120 region. 

-[Ozone distribution analysis](GEMS_TCO_EDA/will_use/TCO_VS_Staratoshere.ipynb): I compared ```total column ozone``` with ```stratospheric ozone``` and analyzed the time series of means and variances of ozone values over time to confirm that most of the ozone is located in the stratosphere. 

## Reference

-[Relevant Gaussian Process related exercises can be found here. ](https://github.com/cl20813/Gaussian_Process_Exercises)      

## Debugging errors 

-[Common errors](errors.md) 




