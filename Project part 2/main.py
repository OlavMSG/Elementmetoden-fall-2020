# -*- coding: utf-8 -*-
"""
Created on 19.11.2020

@author: Olav Milian
"""
import numpy as np
from plotingfunctions import contourplot_Heat2D, meshplot_v2, plotError, plottime
from Heat2D import ThetaMethod_Heat2D
from ErrorEstimate import get_error_estimate
from GradientRecoveryOperator import get_error_estimate_G

def printinfo():
    print("0: End Program")
    print("1: Test time integration method")
    print("2: Test different boundary condition")
    print("3: Test effect of beta")
    print("4: Test effect of a")
    print("5: Make relative error plots")
    print("6: Error estimate using the recovered gradient operator")
    print("7: Make meshplots")

def get_choice():
    printinfo()
    choices = (0, 1, 2, 3, 4, 5, 6, 7)
    choice = -1
    while choice not in choices:
        choice = int(input("Do (0-9): "))
        if choice not in choices:
            print(choice, "is not a valid choice, must be (0-9).")
    return choice


def main():
    f = lambda x, y, t, beta=5: np.exp(- beta * (x * x + y * y))

    fa = lambda x, y, t, beta, a=2: np.exp(- beta * ((x - a * np.sin(t)) ** 2 + y * y))

    uD = lambda x, y, t: np.zeros_like(x)

    uDy = lambda x, y, t: y / 2 + 1 / 2

    duDdt = lambda x, y, t: np.zeros_like(x)

    u0 = lambda x, y: np.zeros_like(x)

    u0y = lambda x, y: y / 2 + 1 / 2

    # general constants
    N = 16
    Nt = 34
    alpha = 9.62e-5
    beta = 3

    choice = -1
    while choice != 0:
        choice = get_choice()

        if choice == 1:
            print("Testing time integration method")
            N = 25
            Nt = 16  # Deltat is smaller than h = 1/N
            alpha = 9.62e-5
            beta = 2

            f1 = lambda x, y, t, beta: fa(x, y, t, beta, a=2)

            save_name = "IntMetTestFE000a001"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f1, uD, duDdt, u0, theta=0, T=2 * np.pi, f_indep_t=False)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)
            save_name = "IntMetTestITrap000a001"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f1, uD, duDdt, u0, theta=0.5, T=2 * np.pi, f_indep_t=False)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)
            save_name = "IntMetTestBE000a001"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f1, uD, duDdt, u0, theta=1, T=2 * np.pi, f_indep_t=False)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)

            Nt = 100  # 2 * N is enough, but we take a big number to see that the methods give about the same solution.
            save_name = "IntMetTestFE000a001smalldt"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f1, uD, duDdt, u0, theta=0, T=2 * np.pi, f_indep_t=False)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)
            save_name = "IntMetTestITrap000a001smalldt"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f1, uD, duDdt, u0, theta=0.5, T=2 * np.pi, f_indep_t=False)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)
            save_name = "IntMetTestBE000a001smalldt"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f1, uD, duDdt, u0, theta=1, T=2 * np.pi, f_indep_t=False)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)

        elif choice == 2:
            print("Testing different boundary condition")
            N = 25
            Nt = 54
            alpha = 9.62e-5
            beta = 2

            save_name = "DiffBCITrapy0y"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f, uDy, duDdt, u0y, theta=0.5, T=1, f_indep_t=True)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)


        elif choice == 3:
            print("Testing effect of beta")
            N = 25
            Nt = 54
            alpha = 9.62e-5

            save_name = "EffectBetaITrap000beta7"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, 7, f, uD, duDdt, u0, theta=0.5, T=1, f_indep_t=True)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)

            save_name = "EffectBetaITrap000beta2"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, 2, f, uD, duDdt, u0, theta=0.5, T=1, f_indep_t=True)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)

        elif choice == 4:
            print("Testing effect of a")
            N = 25
            Nt = 54
            alpha = 9.62e-5
            beta = 2

            save_name = "effectaITrap000a2"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, fa, uD, duDdt, u0, theta=0.5, T=2 * np.pi, f_indep_t=False)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)

            f1 = lambda x, y, t, beta: fa(x, y, t, beta, a=5)
            save_name = "effectaITrap000a5"
            u_hdict = ThetaMethod_Heat2D(N, Nt, alpha, beta, f1, uD, duDdt, u0, theta=0.5, T=2 * np.pi, f_indep_t=False)
            contourplot_Heat2D(N, u_hdict, save_name, save=save)

        elif choice == 5:
            print("Making Relative error plots")
            Nt = 100
            N_list, error_dict, time_vec1, time_stamps = get_error_estimate(fa, uD, duDdt, u0, Nt)
            plotError(N_list[:-1], error_dict, time_stamps, "Rel_tdep", "Relative Error", save=save)
            plottime(N_list, "Rel_tdep", time_vec1, save=save)

            N_list, error_dict, time_vec1, time_stamps = get_error_estimate(f, uD, duDdt, u0, Nt,
                                                                            T=1, f_indep_t=True)
            plotError(N_list[:-1], error_dict, time_stamps, "Rel_tin_dep", "Relative Error", save=save)
            plottime(N_list, "Rel_tin_dep", time_vec1, save=save)

        elif choice == 6:
            print("Error estimate using the Recovered Gradient operator")
            Nt = 100
            N_list, error_dict, time_vec1, time_vec2, time_stamps = get_error_estimate_G(fa, uD, duDdt, u0, Nt)
            plotError(N_list, error_dict, time_stamps, "RGO_tdep", "RGO.Error", save=save)
            plottime(N_list, "RGO_tdep", time_vec1, time_vec2, save=save)

            N_list, error_dict, time_vec1, time_vec2, time_stamps = get_error_estimate_G(fa, uD, duDdt, u0, Nt,
                                                                                         T=1, f_indep_t=True)
            plotError(N_list, error_dict, time_stamps, "RGO_tin_dep", "RGO.Error", save=save)
            plottime(N_list, "RGO_tin_dep", time_vec1, time_vec2, save=save)

        elif choice == 7:
            print("Making meshplots")
            # list of N to plot for
            N_list = [1, 2, 4, 8]
            # make a meshplot
            meshplot_v2(N_list, save=save)
        else:
            # end loop
            break


# save the plot as pdf???
save = True

# run main function
main()





