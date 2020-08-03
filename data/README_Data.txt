Data Archive for the paper
"Learning step-size adaptation in CMA-ES"
-CSA_Data (Contains the performance data of running CMA-ES with CSA)
-	-CSA_Plots_XD (X={5,10,15,...,60} Contains CSA data for functions of dimensionality 5-60D)
-	-CSA_Plots_X (X={50, 100, 150, 200, 250, 500, 1000} Contains CSA data for CMA-ES runs of 50-1000 generations)
-LTO_Data (Contains the saved policies from the runs of CMA-ES with the learned policies for step-size adaptation)
-	-Sampling_Rate_Ablation (Contains the performance data of LTO with sampling rate 0-0.9)
-	-	-Sampling 0.X (X = {0,1,...,9})
-	-Train_5-30D_to_35-60D (Contains the train and test performance data of LTO trained on functions of 5-30D, tested on 35-60D)
-	-Transfer_Other_Fcns (Contains the performance data of LTO trained on 10 BBOB functions, tested on 12 different BBOB functions)
-	-Transfer_Longer_Traj (Contains the performance data of LTO trained on CMA-ES runs of 50 generations, tested on CMA-ES runs of 50-1000 generations)
-	-Transfer5-30D_from10D (Contains the performance data of LTO trained on functions of 10D, tested on 5-30D)