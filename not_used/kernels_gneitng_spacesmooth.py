        ''' 
        # gneiting model (gneting 2002)
        https://www.tandfonline.com/doi/abs/10.1198/016214502760047113

        See the equation (14) on page 5/12.

        $C(s,t) = \frac{\sigma^2}{ (a|t|^{2\alpha}+1 )^{\tau}} \exp(  \frac{ -c||s||^{2\gamma}}{(a|t|^{2\alpha}+1 )^{\beta \gamma}})$, 

        where 
        $s \in R^2$ is spatial distance and $t \in R^1$ is temporal distance.   
        a: scaling parameter of time, non-negative   
        c: scaling parameter of space, non-negative   
        $\alpha, \gamma$: smooth parameter of time, and space. both $ \alpha, \gamma \in (0,1]$      
        $\beta, \tau$: space and time interaction parameters. $\tau >=d/2 = 1$, $\beta \in [0,1]$.  
        '''

    def gneiting_cov_yx(self, params: Tuple[float, float, float, float, float, float, float], y: np.ndarray, x: np.ndarray) -> np.ndarray:
        a, c, alpha, gamma, tau, beta, sigma  = params                 ### x for just consistency with other functions
        nugget = 0.1

        if y is None or x is None:
            raise ValueError("Both y and x_df must be provided.")
     
        # Extract values
        x1 = x[:, 0]
        y1 = x[:, 1]
        t1 = x[:, 3]

        x2 = y[:, 0]
        y2 = y[:, 1]
        t2 = y[:, 3] # hour

        # Efficient distance computation using cdist (operation is vectorized)
        coords1 = np.stack((x1, y1), axis=-1)
        coords2 = np.stack((x2, y2), axis=-1)

        s_dist = cdist(coords1, coords2, 'euclidean')
        t_dist = cdist(t1[:, None], t2[:, None], 'euclidean')  # Ensure t is a 2D array for cdist
        
        # Compute the covariance matrix
        tmp1 = sigma**2 / (a * t_dist**(2 * alpha) + 1)**tau
        tmp2 = np.exp(-c * s_dist**(2 * gamma) / (a * t_dist**(2 * alpha) + 1)**(beta * gamma))
        out = tmp1 * tmp2

        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget
        return out


class space_smooth_experiment:               #sigmasq range advec beta  nugget
    def __init__(self, smooth, input_map, nns_map, mm_cond_number):
        self.smooth = smooth
        self.input_map = input_map
        self.key_list = sorted(input_map)
        self.number_of_timestamps = len(self.key_list)

        sample_df = input_map[self.key_list[0]]

        self.size_per_hour = len(sample_df)
        self.mm_cond_number = mm_cond_number
        nns_map = list(nns_map) # nns_map is ndarray this allows to have sub array of diffrent lengths
        for i in range(len(nns_map)):  
            # Select elements up to mm_cond_number and remove -1
            tmp = np.delete(nns_map[i][:self.mm_cond_number], np.where(nns_map[i][:self.mm_cond_number] == -1))
            if tmp.size>0:
                nns_map[i] = tmp
            else:
                nns_map[i] = []
        self.nns_map = nns_map

    # Custom distance function for cdist
    def custom_distance(self,u, v):
        d = np.dot(self.sqrt_range_mat, u[:2] - v[:2] ) # Distance between x1,x2 (2D)
        spatial_diff = np.linalg.norm(d)  # Distance between x1,x2 (2D)
        temporal_diff = np.abs(u[2] - v[2])           # Distance between y1 and y2
        return np.sqrt(spatial_diff**2 + temporal_diff**2)
    
    def matern_cov_yx(self, sigmasq, params, y_df, x_df) -> pd.DataFrame:
    
        range_lat, range_lon, advec, beta, nugget = params[1:]
             
        # Validate inputs
        if y_df is None or x_df is None:
            raise ValueError("Both y and x_df must be provided.")
        # Extract values
        x1 = x_df['Longitude'].values
        y1 = x_df['Latitude'].values
        t1 = x_df['Hours_elapsed'].values
 
        x2 = y_df['Longitude'].values
        y2 = y_df['Latitude'].values
        t2 = y_df['Hours_elapsed'].values

        spat_coord1 = np.stack((x1- advec*t1, y1 - advec*t1), axis=-1)
        spat_coord2 = np.stack((x2- advec*t2, y2 - advec*t2), axis=-1)

        coords1 = np.hstack ((spat_coord1, (beta * t1).reshape(-1,1) ))
        coords2 = np.hstack ((spat_coord2, (beta * t2).reshape(-1,1) ))

        sqrt_range_mat = np.diag([ 1/range_lon**0.5, 1/range_lat**0.5])
        self.sqrt_range_mat = sqrt_range_mat

        distance = cdist(coords1,coords2, metric = self.custom_distance)

        # Initialize the covariance matrix with zeros
        out = distance
        
        # Compute the covariance for non-zero distances
        non_zero_indices = distance != 0
        if np.any(non_zero_indices):
            out[non_zero_indices] = (sigmasq * (2**(1-self.smooth)) / gamma(self.smooth) *
                                    (distance[non_zero_indices] )**self.smooth *
                                    kv(self.smooth, distance[non_zero_indices]))
        out[~non_zero_indices] = sigmasq

        # Add a small jitter term to the diagonal for numerical stability
        out += np.eye(out.shape[0]) * nugget

        return pd.DataFrame(out)

    def vecchia_likelihood(self, sigmasq, params ,input_df, mm_cond_number, nns_map):

        # reordered_df['ColumnAmountO3'] = reordered_df['ColumnAmountO3']-np.mean(reordered_df['ColumnAmountO3'])
        neg_log_lik = 0
        ## likelihood for the first 30 observations
        # smallset = input_df.iloc[:31,:]
        # neg_log_lik += self.full_likelihood(params, smallset, smallset['ColumnAmountO3'])

        for i in range(0,len(input_df)):
            # current_data and conditioning data
            current_data = input_df.iloc[i:i+1,:]
            current_y = current_data['ColumnAmountO3'].values[0]
            mm_past = nns_map[i,:mm_cond_number]
            mm_past = mm_past[mm_past!=-1]
            # mm_past = np.arange(i)

            conditioning_data = input_df.loc[mm_past,: ]
            df = pd.concat( (current_data, conditioning_data), axis=0)
            y_and_neighbors = df['ColumnAmountO3'].values
            cov_matrix = self.matern_cov_yx(sigmasq, params, y_df= df, x_df=df)
 
            cov_xx = cov_matrix.iloc[1:,1:].reset_index(drop=True)
            cov_yx = cov_matrix.iloc[0,1:]

            # get mean
            locs = np.array(df[['Latitude','Longitude']])

            tmp1 = np.dot(locs.T, np.linalg.solve(cov_matrix, locs))
            tmp2 = np.dot(locs.T, np.linalg.solve(cov_matrix, y_and_neighbors))
            beta = np.linalg.solve(tmp1, tmp2)

            mu = np.dot(locs, beta)
            mu_current = mu[0]
            mu_neighbors = mu[1:]
            
            # mean and variance of y|x
            sigma = cov_matrix.iloc[0,0]
            cov_ygivenx = sigma - np.dot(cov_yx.T,np.linalg.solve(cov_xx, cov_yx))
            cond_mean = mu_current + np.dot(cov_yx.T, np.linalg.solve( cov_xx, (y_and_neighbors[1:]-mu_neighbors) ))   # adjust for bias, mean_xz should be 0 which is not true but we can't do same for y1 so just use mean_z almost 0
            # print(f'cond_mean{mean_z}')

            alpha = current_y - cond_mean
            quad_form = alpha**2 *(1/cov_ygivenx)
            log_det = np.log(cov_ygivenx)
            # Compute the negative log-likelihood

            neg_log_lik += 0.5 * (1 * np.log(2 * np.pi) + log_det + quad_form)
        
        return neg_log_lik


    def mle_parallel(self, key, bounds, sigmasq, initial_params, input_df, mm_cond_number, nns_map):
        try:
            logging.info(f"fit_space_sigma, time: {key}")
            print(f"fit_space_sigma, time: {key}")  # Debugging line
        
            result = minimize(
                self.vecchia_likelihood, 
                sigmasq, 
                args=(initial_params,input_df, mm_cond_number, nns_map),  # neg_ll_nugget(params, input_df, mm_cond_number, ord, nns_map)
                bounds=bounds,
                method='L-BFGS-B'
            )
            jitter = result.x
            logging.info(f"Estimated sigma on {key}, is : {jitter}, when cond {mm_cond_number}, bounds={bounds}")
        
            return f"Estimated sigma on {key}, is : {jitter}, when cond {mm_cond_number},  bounds={bounds}"
        except Exception as e:
            print(f"Error occurred on {key}: {str(e)}")
            logging.error(f"Error occurred on {key}: {str(e)}")
            return f"Error occurred on {key}"
        
