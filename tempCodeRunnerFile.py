    # noise sources matrix N
    def matrix_N(n_abs):
        X = np.zeros((n_abs + 7, n_abs + 4), dtype=np.complex128)  # initialize matrix
        for i in range(n_abs + 7):
            if i == 0:  # johnson Noise (TES1)
                for j in range(n_abs + 4):
                    if j == 0:#[0,0]
                        X[i, j] = -enj / L#
                    elif j == 1:#[1,0]
                        X[i, j] = I * enj / C_tes#

            elif i == 1:  # johnson Noise (Load1)
                for j in range(n_abs + 4):
                    if j == 0:#[0,1]
                        X[i, j] = enj_R / L#
                        break

            elif i == 2:  # Phonon Noise (TES1-Bath)
                for j in range(n_abs + 4):
                    if j == 1:#[1,2]
                        X[i, j] = ptfn_tes_bath / C_tes#
                        break

            elif i == 3:  # Phonon Noise (TES1-Absorber)
                for j in range(n_abs + 4):
                    if j == 1:#[1,3]
                        X[i, j] = ptfn_abs_tes / C_tes#
                    elif j == 2:#[2,3]
                        X[i, j] = -ptfn_abs_tes / C_abs#

            elif i == n_abs + 3:  # Phonon Noise (TES2-Absorber)
                for j in range(n_abs + 4):
                    if j == n_abs + 1:#[2,4]
                        X[i, j] = -ptfn_abs_tes / C_abs#
                    elif j == n_abs + 2:#[3,4]
                        X[i, j] = ptfn_abs_tes / C_tes#

            elif i == n_abs + 4:  # Phonon Noise (TES2-Bath)
                for j in range(n_abs + 4):
                    if j == n_abs + 2:#[3,5]
                        X[i, j] = ptfn_tes_bath / C_tes
                        break

            elif i == n_abs + 5:  # johnson Noise (Load2)
                for j in range(n_abs + 4):
                    if j == n_abs + 3:#[4,6]
                        X[i, j] = enj_R / L
                        break

            elif i == n_abs + 6:  # johnson Noise (TES2)
                for j in range(n_abs + 4):
                    if j == n_abs + 3:#[4,7]
                        X[i, j] = -enj / L
                    elif j == n_abs + 2:#[3,7]
                        X[i, j] = I * enj / C_tes

            #else:  # Phonon Noise (Absorber-Absorber)
                #for j in range(n_abs + 4):
                   # if j == i - 2:
                      #  X[i, j] = ptfn_abs_abs / C_abs
                    #elif j == i - 1:
                        #X[i, j] = -ptfn_abs_abs / C_abs
        return X