Our project part 2 consists of the following files
TMA4220_Project_part2_studentnummers_480858_and_487802.pdf  : The repport.
TMA4220_Project_1_studentnummers_480858_and_487802.pdf  : The previous repport.
readme.txt                 				: What you are reading now.
getplate.py                 		    : Mesh code from https://wiki.math.ntnu.no/tma4220/2020h/project given to us.
Gauss_quadrature.py        				: Script for doing 1D, 1D lineintegral and 2D integrals using Gaussian quadrature.
plotingfunctions.py 				    : Script containing the functions to plot meshes, contourplots of the numerical solution,
                                          plots for the error and plots for the time usage.
Heat2D.py               				: Script to solve the 2D Heat equation using the ThetaMethod.
ErrorEstimate.py                        : Script to make a error estimate
GradientRecoveryOperator.py             : Script to make a error estimate using the Gradient recovery operator
main.py                    				: The main script to control every thing from.

To use the program run the main.py script. 
The Parameters are:
Parameters
----------
N_list : list of ints
    list of different N-s.
	N : int
    	Number of nodes in the mesh.
f : function pointer
    pointer to function to be equal to on the right hand side.
u_exact : function pointer
    Pointer to the function for the exact solution.
g_N : function pointer
     pointer to function for the Neumann B.C.
g_D : function pointer
    pointer to function for the Dirichlet B.C.
DoSingularityCheck : bool, optional
    Call the singularity_check function to check i A is singular before implementation of B.C. The default is True.
save : bool, optional
    Will the plot be saved in the plot folder. The default is False.
    Note: the plot folder must exist!