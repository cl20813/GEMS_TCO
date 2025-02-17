
# Full Likelihood V.S. Vecchia Approximated Likelihood

### Data: ```GEMS July 1st, 2024```, Data size: ```1,250 x 8(hours)```, ```Model: Matern $\biggl(\sqrt{ ||x-vt||+ \beta^2 t^2} \biggr)$ ```.

Conditioning number: 10 (5 on current space, 5 on one lag space)

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [60, 8.25, 8.25, 0.5, 0.5, 5.0]          
```Full Likelihood: 24274.6368``` (826 seconds).     
```Vecchia Approximation: 24312.1206``` (0.0101 seconds, this does not make sense)     

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [40, 5.25, 5.25, 0.5, 0.5, 0.5]           
```Full Likelihood: 24443.6162``` (881 seconds).        
```Vecchia Approximation: 24354.4574``` 

Conditioning number: 20 (10 on current space, 10 on one lag space)

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [60, 8.25, 8.25, 0.5, 0.5, 5.0]          
```Full Likelihood: 24274.6368``` (909 seconds).     
```Vecchia Approximation: 24312.1206``` (0.0101 seconds, this does not make sense)     

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [40, 5.25, 5.25, 0.5, 0.5, 0.5]           
```Full Likelihood: 24443.6162``` (881 seconds).        
```Vecchia Approximation: 24266.6935``` (0.0101 seconds, this does not make sense)   

```Summary```       
Full likelihood change: 24274.6368-24443.6162 = ```-168.9794```         
Vecchia likelihood change conditioning on 10:  24312.1206-24354.4574 = ```-42.3368```       
Vecchia likelihood change conditioning on 20:  24312.1206-24266.6935 = ```45.4271```           

### Data: ```GEMS July 1st, 2023```, Data size: ```1,250 x 8(hours)```, Model: ```Matern($\sqrt{ ||x-vt||+\beta^2t^2}}$)```.

Conditioning number: 10 (5 on current space, 5 on one lag space)

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [60, 8.25, 8.25, 0.5, 0.5, 5.0]          
```Full Likelihood: 24513.3406``` (914 seconds).              
```Vecchia Approximation: 24519.9868``` (0.0101 seconds, this does not make sense)     

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [40, 5.25, 5.25, 0.5, 0.5, 0.5]           
```Full Likelihood: 25074.6417``` (984 seconds).            
```Vecchia Approximation: 24937.278```        

Conditioning number: 20 (10 on current space, 10 on one lag space)

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [60, 8.25, 8.25, 0.5, 0.5, 5.0]          
```Full Likelihood: 24513.3406``` (836 seconds).     
```Vecchia Approximation: 24506.5732``` (0.0101 seconds, this does not make sense)  

parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [40, 5.25, 5.25, 0.5, 0.5, 0.5]           
```Full Likelihood: 25074.6417``` (996 seconds).            
```Vecchia Approximation: 25029.0841```       

```Summary```              
Full likelihood change: 24513.3406- 25074.6417 = ```-561.3011```       
Vecchia likelihood change conditioning on 10:  24519.9868-24937.278 = ```-417.2912```       
Vecchia likelihood change conditioning on 20:  24506.5732- 25029.0841 = ```-522.5109```           


# Model Fitting

### Data: ```GEMS July 2024```, Data size: ```50 x 8(hours)```, Model: ```Matern($\sqrt{ ||x-vt||+\beta^2t^2}}$)```.



#### July 1st 2024: Conditioning number: 10 (5 on current space, 5 on one lag space)         
Initial parameters ```(sigmasq, range_latitude, range_longitude, advection, beta and nugget)```: [40, 5.25, 5.25, 0.3, 0.3, 0.5],  bounds=[(0.05, 40.0), (0.05, 15.0), (0.05, 15.0), (-15.0, 15.0), (0.25, 20.0), (0.05, 0.5)], smooth=0.5        
```Estimates from Full Likelihood:          ```[31.38 4.61 3.36 9.26e-04 0.25 0.05]  390.75 seconds (40 iterations)```                     
```Estimates from Vecchia Approximation:    ```[11.73 0.05 0.05 -0.10 0.25 0.055  ]  291.68 seconds (66 iterations)```        
         
#### July 2nd 2024:  Conditioning number: 10 (5 on current space, 5 on one lag space)       
Initial parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [40, 5.25, 5.25, 0.3, 0.3, 0.5],  bounds=[(0.05, 40.0), (0.05, 15.0), (0.05, 15.0), (-15.0, 15.0), (0.25, 20.0), (0.05, 0.5)], smooth=0.5        
```Estimates from Full Likelihood:          ```[24.88 6.83 8.33 0.0157 0.25 0.05]```  385.65 seconds (34 iterations)                        
```Estimates from Vecchia Approximation:    ```[16.20 2.78 7.55 -0.235 0.25 0.5 ]```  164.55 seconds (51 iterations)      
    
 Conditioning number: 20 (10 on current space, 10 on one lag space)   
Initial parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [40, 5.25, 5.25, 0.3, 0.3, 0.5],  bounds=[(0.05, 40.0), (0.05, 15.0), (0.05, 15.0), (-15.0, 15.0), (0.25, 20.0), (0.05, 0.5)], smooth=0.5     
```Estimates from Full Likelihood:          ```[24.88 6.83 8.33 0.0157 0.25 0.05]``` 385.65 seconds (34 iterations)                   
```Estimates from Vecchia Approximation:    ```[20.17 4.27 6.58 -0.159 0.25 0.05]```  355.72 seconds (36 iterations)

#### July 3rd 2024:  Conditioning number: 10 (5 on current space, 5 on one lag space)       
Initial parameters (sigmasq, range_latitude, range_longitude, advection, beta and nugget): [40, 5.25, 5.25, 0.3, 0.3, 0.5],  bounds=[(0.05, 40.0), (0.05, 15.0), (0.05, 15.0), (-15.0, 15.0), (0.25, 20.0), (0.05, 0.5)], smooth=0.5        
```Estimates from Full Likelihood:          ```[24.88 6.83 8.33 0.0157 0.25 0.05]```  385.65 seconds (34 iterations)                        
```Estimates from Vecchia Approximation:    ```[16.20 2.78 7.55 -0.235 0.25 0.5 ]```  164.55 seconds (51 iterations)    
 





