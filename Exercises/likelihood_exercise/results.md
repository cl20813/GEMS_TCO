
# Full Likelihood V.S. Vecchia Approximated Likelihood

### Data: ```GEMS July 1st, 2024```, Data size: ```1,250 x 8(hours)```, Model: Matern ( $\sqrt{||x-vt|| + \beta^2 t^2}$ ).

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

### Data: ```GEMS July 2024```, Model: ```Matern($\sqrt{ ||x-vt||+\beta^2t^2}}$)```.
Initial parameters ```(sigmasq, range_latitude, range_longitude, advection, beta and nugget)```: [40, 5.25, 5.25, 0.3, 0.3, 0.5],  bounds=[(0.05, 40.0), (0.05, 15.0), (0.05, 15.0), (-15.0, 15.0), (0.25, 20.0), (0.05, 0.5)], smooth=0.5  

#### July 1st 2024: Conditioning number: 10 (5 on current space, 5 on one lag space)    
Data size: ```50 x 8(hours)```
```Estimates from Full Likelihood:          [31.38 4.61 3.36 9.26e-04 0.25 0.05] 390.75 seconds (40 iterations)```                      
```Estimates from Vecchia Approximation:    [11.73 0.05 0.05 -0.10 0.25 0.055  ] 291.68 seconds (66 iterations)```     

Data size: ```100 x 8(hours)```
```Estimates from Full Likelihood:          [36.99 2.72 2.37 0.07 0.25 0.5] 892.48 seconds (28 iterations)```                      
```Estimates from Vecchia Approximation:    [11.98 0.05 0.05 -0.02 0.95 0.5] 460.82 seconds (60 iterations)``` 

Data size: ```200 x 8(hours)```
```Estimates from Full Likelihood:          [28.55 1.68 1.58 -0.06 0.25 0.5] 5698.98 seconds (46 iterations)```                      
```Estimates from Vecchia Approximation:    [8.98 0.05 0.05 -0.04 0.98 0.5] 913.02 seconds (58 iterations)```

         
#### July 2nd 2024:  Conditioning number: 10 (5 on current space, 5 on one lag space)       

```Estimates from Full Likelihood:          [24.88 6.83 8.33 0.016 0.25 0.05] 385.65 seconds (34 iterations)```                           
```Estimates from Vecchia Approximation:    [16.20 2.78 7.55 -0.24 0.25 0.5 ] 164.55 seconds (51 iterations)```          
 Conditioning number: 20 (10 on current space, 10 on one lag space)                 
```Estimates from Vecchia Approximation:    [20.17 4.27 6.58 -0.16 0.25 0.05] 355.72 seconds (36 iterations)```   

Data size: ```100 x 8(hours)```
```Estimates from Full Likelihood:          [22.27 3.96 3.06 -2.99e-03 0.25 0.5] 1477.67 seconds (44 iterations)```                      
```Estimates from Vecchia Approximation:    [6.95 0.05 0.05 -0.13 0.25 0.5] 508.41 seconds (67 iterations)```

Data size: ```200 x 8(hours)```
```Estimates from Full Likelihood:          [19.69 2.18 2.08 -0.02 0.25 0.5] 5325.9134 seconds (40 iterations)```                      
```Estimates from Vecchia Approximation:    [5.68 0.05 0.05 5.05ㄷ-04 1.11 0.5] 981.91 seconds (51 iterations)```

#### July 3rd 2024:  Conditioning number: 10 (5 on current space, 5 on one lag space)       
    
```Estimates from Full Likelihood:          [35.87 3.09 2.97 0.08 0.25 0.5  ] 365.65 seconds (40 iterations)```                           
```Estimates from Vecchia Approximation:    [13.81 0.05 0.05 0.02 0.79 0.06 ] 216.11 seconds (60 iterations)``` 

Data size: ```100 x 8(hours)```
```Estimates from Full Likelihood:          [34.57 2.35 2.43 -0.10 0.25 0.5] 1068.67 seconds (32 iterations)```                      
```Estimates from Vecchia Approximation:    [12.75 0.05 0.05 -0.14 0.25 0.06 ] 300.90 seconds (36 iterations)```

Data size: ```200 x 8(hours)```
```Estimates from Full Likelihood:          [32.52 1.85 1.47 -0.09 0.25 0.5] 4455.90 seconds (36 iterations)```                      
```Estimates from Vecchia Approximation:    [10.67 0.05 0.05 -0.07 0.92 0.5] 801.82 seconds (47 iterations)```

# Intial parameters search

1 10 10 5 5 0.01 8175   
1 10 10 5 5 0.5 3207   

1 10 10 0.1 0.1 0.5 2810   

20 1 1 0.5 0.1 0.01 1052.09   
20 1 1 0.5 0.1 0.05 1052.192   
20 1 1 0.5 0.1 0.5 1053   

10 1 1 0.1 0.5 0.5 1038   

20 1 1 0.1 0.5 0.5 1029   
20 1 1 0.1 0.5 0.01 1026.73   


# Model Fitting (updated Vecchia computation time complexity)

### Data: ```GEMS July 2024```, Model: ```Matern($\sqrt{ ||x-vt||+\beta^2t^2}}$)```.
Initial parameters ```(sigmasq, range_latitude, range_longitude, advection, beta and nugget)```: [20, 1, 1, 0.1, 0.5, 0.1],  bounds=[(0.05, 40.0), (0.05, 15.0), (0.05, 15.0), (-15.0, 15.0), (0.25, 20.0), (0.05, 0.5)], smooth=0.5  

#### July 1st 2024: Conditioning number: 10 (5 on current space, 5 on one lag space)    
Data size: ```50 x 8(hours)```
```Estimates from Full Likelihood:          [31.38 4.61 3.36 9.26e-04 0.25 0.05] 553.19 seconds (55 iterations)```                      
```Estimates from Vecchia Approximation:    [11.73 0.05 0.05 -0.10 0.25 0.055  ] 41.4231 seconds (44 iterations)```     

Data size: ```100 x 8(hours)```
```Estimates from Full Likelihood:          [36.77 2.68 2.34 0.07 0.25 0.5] 1449.6128 seconds (28 iterations)```                      
```Estimates from Vecchia Approximation:    [11.98 0.05 0.05 -0.02 0.95 0.5] 106.5434 seconds (52 iterations)``` 

Data size: ```200 x 8(hours)```
```Estimates from Full Likelihood:          [28.35 1.66 .155 -0.06 0.25 0.5] 4864.7468 seconds (29 iterations)```                      
```Estimates from Vecchia Approximation:    [8.98 0.05 0.05 -0.04 0.98 0.5] 138.5727 seconds (34 iterations)```

         
#### July 2nd 2024:  Conditioning number: 10 (5 on current space, 5 on one lag space)       

```Estimates from Full Likelihood:          [24.75 6.74 8.23 0.016 0.25 0.05] 394.22 seconds (39 iterations)```                               
```Estimates from Vecchia Approximation:    [16.14 2.74 7.47 -0.23 0.25 0.05] 30.35 seconds (37 iterations)```   

Data size: ```100 x 8(hours)```
```Estimates from Full Likelihood:          [22.09 3.89 3.01 -2.64e-03 0.25 0.5] 1445.4218 seconds (40 iterations)```                      
```Estimates from Vecchia Approximation:    [6.63 0.11 0.10 -0.20 0.25 0.49] 95.73 seconds (35 iterations)```

Data size: ```200 x 8(hours)```
```Estimates from Full Likelihood:          [19.69 2.15 2.05 -0.017 0.25 0.5] 5305.00 seconds (30 iterations)```                      
```Estimates from Vecchia Approximation:    [5.68 0.05 0.05 3.79e-04 1.11 0.5] 181.29 seconds (48 iterations)```

#### July 3rd 2024:  Conditioning number: 10 (5 on current space, 5 on one lag space)       
    
```Estimates from Full Likelihood:          [35.72 3.06 2.93 0.07 0.25 0.5] 336.57 seconds (28 iterations)```                           
```Estimates from Vecchia Approximation:    [13.38 0.05 0.05 0.11 0.25 0.5] 34.34 seconds (34 iterations)``` 

Data size: ```100 x 8(hours)```
```Estimates from Full Likelihood:          [34.36 2.33 2.40 -0.09 0.25 0.5] 1352.07 seconds (32 iterations)```                      
```Estimates from Vecchia Approximation:    [12.30 0.05 0.05 -0.13 0.25 0.5] 122.58 seconds (48 iterations)```

Data size: ```200 x 8(hours)```
```Estimates from Full Likelihood:          [32.33 1.83 1.45 -0.087 0.25 0.5] 5110.91 seconds (29 iterations)```                      
```Estimates from Vecchia Approximation:    [10.67 0.05 0.05 -0.069 0.91 0.5] 131.74 seconds (32 iterations)```





