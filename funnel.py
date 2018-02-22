import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------#
# MSAN603 FUNNEL SIMULATION                                                          	 #
# Members:  Ker-Yu Ong - Maise Ly - Shikhar Gupta - Sooraj Subrahmannian - Yiqiang Zhao  #		
#----------------------------------------------------------------------------------------#


## HELPER FUNCTIONS ##

#----------------------------------------------------------------------
# QUESTION 1
#----------------------------------------------------------------------
def UserSim(n, lmbd):
    """
    Input: 
    - n: number of users to stimulate
    - lmbd: parameter
    
    Output: 
    - list of exponential random variable simulations (not array)
    """
    return list(np.random.exponential(scale=1.0/lmbd, size=n))


def plot_funnel(stoptimes, start, end, inv, title):
    """
    Plot funnel graph given users' quit times and bins
    """
    stoptimes = np.array(stoptimes)
    bins = np.arange(start, end+inv, inv)
    funnel_bins = [sum(stoptimes >= edge) for edge in bins]
    
    plt.bar(bins, funnel_bins, width=.2)
    plt.ylabel('The number of users who survive')
    plt.xlabel('Stop time')
    plt.title(title)


#----------------------------------------------------------------------
# QUESTION 2
#----------------------------------------------------------------------
def EstLam1(user_lst):
    """
    Return the MLE of lambda using user list
    """
    x_avg = np.array(user_lst).mean()
    est_lambda = 1/x_avg
    return est_lambda


def lambdaCI(stoptimes, n_bootstraps=500, alpha=0.95):
    """
    Input: 
    - stoptimes: list of users' stoptimes 
    - n_bootstraps: number of bootstraps
    - alpha: 1 - confidence level
    
    Output: (1-alpha/2), alpha/2 percentile of bootstrap
    """
    est_lmbd = EstLam1(stoptimes)
    est_lambdas = []
    num_users = len(stoptimes)
    
    # boostrap from the given user quit times
    for i in range(n_bootstraps):
        new_stoptimes = np.random.choice(stoptimes, num_users)
        new_est_lambda = EstLam1(new_stoptimes)
        est_lambdas.append(new_est_lambda)
    
    # compute confidence interval for the boostrapped lambda estimates
    est_lambdas = np.asarray(est_lambdas)
    upper_p, lower_p = (1 - (alpha / 2)) * 100, (alpha / 2) * 100
    upper = np.percentile(est_lambdas, upper_p)
    lower = np.percentile(est_lambdas, lower_p)
    
    return (est_lmbd, lower, upper)

#----------------------------------------------------------------------
# QUESTION 4
#----------------------------------------------------------------------
def HurdleFun(stoptimes, breaks):
    """
    Input: 
    - List from UserSim of times that users quit (user quit times)
    - List of breakpoints (of arbitrary length)
    Output:
    - List of the number of users who fail to proceed beyond that break-point, 
      but did get to the previous break-point.
    """
    brks = [0] + breaks; brks = brks + [999e99] # add zero and a large number to the beginning and end of the list
    output = [ sum( [ 1 for t in stoptimes if t < brks[i] and t >= brks[i-1] ] )   
             for i, n in enumerate(brks) ][1:]
    return output


def cdf(x, lmbd): 
	"""
	Compute the CDF of an exponentially distributed variable
	"""
	return 1 - np.exp(-lmbd * x)


def EstLam2(hurdle_list, breaks):
    """
    Input: 
    - Hurdle output list
    - List of breakpoints, which is the same input of HurdleFun
    
    Output: 
    - A python lambda function which takes, as its sole input, a lambda value 
    and which will return the log likelihood value for the data originally entered in HurdleFun. 
    """
    m0, m2 = hurdle_list[0], hurdle_list[-1]
    
    return lambda lmbd: sum([np.log( np.exp(-lmbd*breaks[i]) - np.exp(-lmbd*breaks[i+1]) ) * hurdle_list[i+1] \
    						for i in range(len(breaks)-1)]) + \
		                m0 * np.log( 1 - np.exp(-lmbd * breaks[0]) ) + \
		                m2 * (-lmbd) * breaks[-1]


def MaxMLE(hurdle_list, breaks, lmbds):
    """
    Input: 
    - List of hurdles 
    - List of breakpoints
    - Range of lambda values
    
    Output: 
    - the lambda that maximizes the likelihood
    
    Example: 
    MaxMLE( HurdleFun(x, breaks), breaks, list(numpy.arange(.1, 3, .05)))
    """
    loglik = EstLam2(hurdle_list, breaks)
    idx = np.argmax([ loglik(lmbd) for lmbd in lmbds ])
    return lmbds[idx]



if __name__ == '__main__':

	#----------------------------------------------------------------------
	# QUESTION 1
	#----------------------------------------------------------------------
	print("\nQUESTION 1")
	
	print("Q1.A")
	np.random.seed(42)
	stoptimes = UserSim(n = 1000, lmbd = 2)
	start, end, inv = 0.0, 3.0, .25
	plt.figure(figsize=(7, 4))
	plot_funnel(stoptimes, start, end, inv, title="Funnel for 1000 users & $\lambda$ = 2")
	plt.savefig("Q1A.png")
	print("Graph was saved to current directory.")
	# plt.show()

	print("\nQ1.B")
	lambda_range = np.arange(0.2, 3.2, .2)
	plt.figure(figsize=(20,30))
	plt.subplots_adjust(hspace=0.4, wspace=0.4)
	for i, lmbd in enumerate(lambda_range): 
	    stoptimes = UserSim(n = 1000, lmbd = lmbd)
	    start, end, inv = 0.0, 3.0, .25
	    plt.subplot(np.ceil(len(lambda_range)/2)+1, 2, i+1)
	    plot_funnel(stoptimes, start, end, inv, title="Funnel for 1000 users & $\lambda$ = %.2f" % lmbd)	
	
	plt.savefig("Q1B.png")
	print("Graph was saved to current directory.")    
	# plt.show()

	#----------------------------------------------------------------------
	# QUESTION 2
	#----------------------------------------------------------------------
	print("\nQUESTION 2")
	print("Q2.B")
	stoptimes = UserSim(n = 1000, lmbd = 1)
	est_lambda = EstLam1(stoptimes)
	print("Estimated Lambda is: %.4f" % est_lambda)

	print("\nQ2.C")
	est_lmbd, lower_bound, upper_bound = lambdaCI(stoptimes, n_bootstraps=500, alpha=0.95)
	print("95% Confidence Interval for the Lambda Estimate: ")
	print("\tLower Bound: %.6f" %  lower_bound)
	print("\tUpper Bound: %.6f" %  upper_bound)

	print("\nQ2.D")
	users_range = [100, 200, 500, 1000, 2000, 5000, 10000]
	lambdaCI_df = pd.DataFrame({
	    'num_users': users_range,
	    'lambda': np.ones(len(users_range),)
	})

	lambdaCI_df['estimated_lambda'], lambdaCI_df['lower_bound'], lambdaCI_df['upper_bound'] = \
                zip(*lambdaCI_df['num_users'].apply(lambda x: lambdaCI(UserSim(x, lmbd = 1), 500, 0.05)))

	lambdaCI_df['CI_range'] = lambdaCI_df['upper_bound'] - lambdaCI_df['lower_bound']
	
	# print output
	print(lambdaCI_df)

	# plot confidence interval
	plt.figure(figsize=(7, 4))
	plt.plot(lambdaCI_df['num_users'], lambdaCI_df['upper_bound'], marker='o', label='Upper Bound')
	plt.plot(lambdaCI_df['num_users'], lambdaCI_df['estimated_lambda'], marker='o', label='Estimated Lambda')
	plt.plot(lambdaCI_df['num_users'], lambdaCI_df['lower_bound'], marker='o' , label='Lower Bound')
	plt.legend(loc='best')
	plt.title("Number of users vs. Estimated Lambda")
	plt.xlabel("Number of users")
	plt.ylabel("Estimated Lambda")
	# plt.show()
	plt.savefig("Q2D.png")
	print("\nGraph was saved to current directory.")


	#----------------------------------------------------------------------
	# QUESTION 4
	#----------------------------------------------------------------------
	print("\nQUESTION 4")
	breakpoints_lst = [[.25, .75], [.25, 3], [.25, 10]]
	est_lmbd, mle_lmbd, diff = [], [], []

	for brks in breakpoints_lst:
		np.random.seed(42)
		for i in range(1000): 
			tmp_est, tmp_mle,tmp_diff = [], [], []   
			# simulate user list
			stoptimes = UserSim(100, 1) 

			# calculate lambda 
			lam1 = EstLam1(stoptimes)
			lam2 = MaxMLE( HurdleFun(stoptimes, brks), brks, list(np.arange(.1, 3, .05)))
			tmp_est.append(lam1); tmp_mle.append(lam2); tmp_diff.append(lam1-lam2)
	    
	    # mean of 1000 simulations
		mean_est, mean_mle,mean_diff = np.mean(tmp_est), np.mean(tmp_mle),np.mean(tmp_diff) 
		est_lmbd.append(mean_est); mle_lmbd.append(mean_mle);diff.append(mean_diff)

	# output
	diff_lmbd = pd.DataFrame({
    'Breakpoints': breakpoints_lst,
    'EstLam1': est_lmbd,
    'EstLam2': mle_lmbd,
    'Diff': diff
	})

	print(diff_lmbd)


	# TO BE REMOVED
	lambdaCI_df.to_csv("lambdaCI.csv") 
	diff_lmbd.to_csv("diff_df.csv")
