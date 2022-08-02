#!/bin/env python3

import os
import glob
import random
import os
import sys
import time
from threading import Thread
import numpy as np
from pickletools import optimize
import tensorflow as tf
from scipy.integrate import simps

# define fitness Function
def FitnessFunction(total_density, AA_density, all_score):
    #print(X)
    x = total_density[:, 0]
    score = all_score[:, 0]
    return (x/AA_density - 1)**2 + score

# update the velocity of particle
def UpdateVelocity(V, X, pbest, gbest, c1, c2, w, max_vel):
    size = X.shape[0]
    r1 = np.random.random((size,1))
    r2 = np.random.random((size,1))
    V = w*V + c1*r1*(pbest-X) + c2*r2*(gbest-X)
    
    # handle the max velocity
    V[V<-max_vel] = -max_vel
    V[V>max_vel] = max_vel

    return V

# update the position of particle
def UpdatePosition(X, V):
    return X+V

# run one MD simulation
def RunMD(i, type=" "):
    os.system("cd MD_run/particle_%s; chmod +x run.sh; %s ./run.sh 1>/dev/null 2>&1; cd ../../" % (str(i+1), type))
    
# run all MD
def RunAllMDLocal(process_num, particle_num):
    process_total = []
    tmp = []
    for i in range(particle_num):
        tmp.append(i)
        if len(tmp) >= process_num or i == particle_num-1:
            process_total.append(tmp)
            tmp = []
    for i in process_total:
        process_list = []
        print("Sub process: %s/%d" %(str([sb+1 for sb in i]), particle_num))
        for sub_process in i:
            p = Thread(target=RunMD, args=(sub_process,))
            p.start()
            process_list.append(p)
        for p_num in process_list:
            p.join()
            
# run all MD to slurm system
def RunAllMDSlurm(process_num, particle_num):
    process_total = []
    tmp = []
    for i in range(particle_num):
        tmp.append(i)
        if len(tmp) >= process_num or i == particle_num-1:
            process_total.append(tmp)
            tmp = []
    for i in process_total:
        process_list = []
        print("Sub process: %s/%d" %(str([sb+1 for sb in i]), particle_num))
        for sub_process in i:
            RunMD(sub_process, "sbatch")
        done_num = 0
        time_start = time.time()
        while True:
            for sub_process in i:
                if os.path.exists("MD_run/particle_%s/tag_finished" % (sub_process+1)) == True:
                    done_num += 1
                    time_end = time.time()
                    time_sum = time_end - time_start
                    print("Process done: %s, Time: %s seconds" % (sub_process+1, round(time_sum,3)))
                    os.system("rm MD_run/particle_%s/tag_finished" % (sub_process+1))
            if done_num == len(i):
                break
           
# get density after MD simulation
def GetDensity(i):
    # get CGMD properties
    process = os.popen("grep 'kg/m^3' MD_run/particle_%s/den.log | awk ' {print $2}' " % str(i+1)) # return density
    density = process.read()
    process.close()
    #print("density: %s" % str(density))
    return density

# judge the digit
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# get avg value of rdf
def GetRDFvalue(atom_type_list,path):
    atom_type_rdf_dict = {}
    rdf_max_position = []
    for CGtype in atom_type_list:
        info11 = CGtype[0]
        info12 = CGtype[1]
        x_arry = []
        y_arry = []
        file_name_total = []
        ######################################
        for file_name1 in glob.glob(str(path)+'/'+str(info11)+'_'+str(info12)+'.xvg'):
            file_name_total.append(file_name1)
        if len(file_name_total) == 0:
            for file_name2 in glob.glob(str(path)+'/'+str(info12)+'_'+str(info11)+'.xvg'):
                file_name_total.append(file_name2)
        ######################################
        #print(file_name)
        x_arry0 = []
        y_arry0 = []
        if len(file_name_total) != 0:
            f1 = open(file_name_total[0],'r')
            #print(f1)
            for info2 in f1:
                info3 = info2.strip('\n')
                x = info3.split()[0]
                if is_number(x) == True:
                    y = info3.split()[1]
                    if float(x) <= 0.8: # RDF's max x value (nm).
                        x_arry0.append(float(x))
                        y_arry0.append(round(float(y),4))
            f1.close()
        else:
            x_arry0 = [i*0.01 for i in range(int(0.8/0.002))]
            y_arry0 = [0.0] * int(0.8/0.002)
        rdf_max_position.append(x_arry0[y_arry0.index(max(y_arry0))])
        #print("rdf_max_position: %s" % rdf_max_position)
        atom_type_rdf_dict[str(info11)+'_'+str(info12) +'_x'] = x_arry0
        atom_type_rdf_dict[str(info11)+'_'+str(info12)+'_y'] = y_arry0
    return atom_type_rdf_dict, rdf_max_position

# defin score function (np.exp(-x) is weight function)
def Score_func(list_x1,list_y1,list_x2,list_y2):
    list_tmp_x = []
    list_tmp_y = []
    tmp = 0
    while tmp <= 0.8:
        tmp = round(tmp,2)
        #print(tmp)
        if (tmp in list_x1) and (tmp in list_x2):
            index1 = list_x1.index(tmp)
            index2 = list_x2.index(tmp)
            y11 = list_y1[index1]
            y22 = list_y2[index2]
            x = tmp
            y = np.exp(-x)*(y11-y22)**2
            list_tmp_x.append(x)
            list_tmp_y.append(round(y,8))
        tmp += 0.01
    integr_num = round(simps(list_tmp_y, list_tmp_x),4)
    return integr_num

# get CG RDF and compare the RDF between AA and CG, and retrun the score function
def GetAllscore(particle_num, atom_type_rdf_AA, CG_rdf_pair):
    all_score = []
    rdf_max_position = []
    for i in range(particle_num):
        # get the value of RDF
        CG_rdf_path = "MD_run/particle_%s" % str(i)
        atom_type_rdf_CG, rdf_max_position_tmp  = GetRDFvalue(CG_rdf_pair, CG_rdf_path)
        # get the max position
        rdf_max_position.append(rdf_max_position_tmp) 
        # compare rdf, retrun the score function
        total_score = 0
        for CGtype in CG_rdf_pair:
            score = Score_func(atom_type_rdf_AA[str(CGtype[0])+'_'+str(CGtype[1]) +'_x'],
                            atom_type_rdf_AA[str(CGtype[0])+'_'+str(CGtype[1]) +'_y'],
                            atom_type_rdf_CG[str(CGtype[0])+'_'+str(CGtype[1]) +'_x'],
                            atom_type_rdf_CG[str(CGtype[0])+'_'+str(CGtype[1]) +'_y'])
            total_score += score
        all_score.append([total_score])          
    return np.array(all_score), np.array(rdf_max_position)

# calculate the core of position of max rdf: sum( (CGRDF/AARDF-1)^2 )
def GetMaxRDFcore(particle_num, AA_rdf_max_position, rdf_max_position):
    AA_rdf_max_position = np.array([AA_rdf_max_position])
    AA_rdf_max_position_extend = np.empty(shape=(0, len(AA_rdf_max_position[0])))
    for i in range(particle_num):
        AA_rdf_max_position_extend = np.r_[AA_rdf_max_position_extend, AA_rdf_max_position] 
    temp = (rdf_max_position / AA_rdf_max_position_extend - 1)**2
    temp = np.sum(temp, axis=1)
    max_RDF_core = []
    for i in range(len(temp)):
        max_RDF_core.append([temp[i]])
    return np.array(max_RDF_core)

# the ANN model
def ANN(x_train, y_train, x_predict, BATCH_SIZE, EPOCHS=1000):
    # the node of hidden
    n_hidden_1 = 50
    n_hidden_2 = 50
    num_input = len(x_train[0])
    num_output = len(y_train[0])
    # create model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden_1, input_dim = num_input, activation='relu'))
    model.add(tf.keras.layers.Dense(n_hidden_2, activation='relu'))
    model.add(tf.keras.layers.Dense(num_output, activation='relu'))
    model.summary()
    
    # set loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['acc'])

    # train model
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    # evaluate the model
    __, accuracy = model.evaluate(x_train, y_train, verbose=0)
    print('Accuracy of ANN model:%.2f %%' % (accuracy*100))

    # predictions with the model
    predictions = model.predict(x_predict, verbose=0)
    #for i in range(len(x_train)):
    #    print('%s => %s (expected %s)' % (x_train[i].tolist(), predictions[i], y_train[i]))
    return predictions

# main function
def main(MDparameter_num, particle_num, iter_num, max_vel, CG_atom_type, AA_density, 
         process_num, sigma_min_max, epsilon_min_max, chaostic_begin, AA_rdf_path, 
         CG_rdf_pair, ANN_begin, EPOCHS, BATCH_SIZE, collect_sets, type_system):
    # algorithm parameter inital setup
    w_init = 0.9
    w_end = 0.4
    c1 = 2
    c2 = 2
    r1 = None
    r2 = None
    u = 4.0
    #
    fitness_val_list = []
    # init position
   
    # Logistic function
    z = np.empty(shape=(0, MDparameter_num))
    z0 = np.random.uniform(0, 1.0, size=(1, MDparameter_num))
    for i in range(particle_num):
        z = np.r_[z, z0]   
        z0 = np.multiply(4*z0, 1-z0)
    # init position
    X = []
    for i in range(particle_num):
        tmp = []
        for j in range(MDparameter_num):
            if j % 2 == 0:
                # transformation
                X_tmp = sigma_min_max[0] + (sigma_min_max[1]-sigma_min_max[0])*z[i][j]
            else:
                # transformation
                X_tmp = epsilon_min_max[0] + (epsilon_min_max[1]-epsilon_min_max[0])*z[i][j]
            tmp.append(X_tmp)
        X.append(tmp)
    X = np.array(X)
    print("***** initial sigma epsilon   *****")
    print(X)
    # init velocitiy
    V = np.random.uniform(0, max_vel, size=(particle_num, MDparameter_num))

    # iteration
    gbest_file = open("gbest_value.dat",'w+')
    gbest_file.write("iter   sigma epsilon sigma epsilon ……\n")

    # get AA MD rdf value
    atom_type_rdf_AA, AA_rdf_max_position = GetRDFvalue(CG_rdf_pair, AA_rdf_path)
    # define ANN parameter
    x_predict = np.array([[AA_density]+AA_rdf_max_position])
    #print(x_predict)
    x_train = np.empty(shape=(0, len(x_predict[0])))
    y_train = np.empty(shape=(0, MDparameter_num))
    #print(x_train.shape)
    #print(y_train)
    
    # begin iteration
    for k in range(iter_num):
        print("!!!!!!!!  running...... iter: %d/%d   !!!!!!!!!!!" % (k+1, iter_num))
        if k==0:
            pbest = X
            # update the sigma and epsilon in the itp file
            for i in range(particle_num):
                itp_file = open("MD_run/particle_%s/atomtypes.itp" % str(i+1), 'w+')
                itp_file.write("[ atomtypes ]\n \
                                ;name   bond_type     mass     charge   ptype   sigma         epsilon       Amb\n")
                jj = 0
                while jj < len(CG_atom_type):
                    itp_file.write("%s      %s      0.00000  0.00000   A   %8.5e      %8.5e \n"  \
                                    % (CG_atom_type[jj], CG_atom_type[jj], X[i][2*jj], X[i][2*jj+1]))
                    jj += 1
                itp_file.close()  

            # run MD simulation
            if type_system == "local":
                RunAllMDLocal(process_num, particle_num)         
            elif type_system == "slurm":
                RunAllMDSlurm(process_num, particle_num)
            else:
                print("!!!!!!!!!!!!!type_system error!!!!!!!!!!!!")
                sys.exit()
            # get density
            total_density = []
            for i in range(particle_num):
                density = GetDensity(i)
                if str(density) == "-nan" or str(density) == "":
                    density = 1000
                total_density.append([float(density)/1000.0])
            total_density = np.array(total_density)
            
            # get score value
            all_score, rdf_max_position = GetAllscore(particle_num, atom_type_rdf_AA, CG_rdf_pair)
            max_RDF_core = GetMaxRDFcore(particle_num, AA_rdf_max_position, rdf_max_position)
            #print("Best score: %f" % float(np.min(all_score)))
            #print("Best score particle num: %d" % int(np.argmin(all_score)+1))
            print("Best score: %f" % float(np.min(max_RDF_core)))
            print("Best score particle num: %d" % int(np.argmin(max_RDF_core)+1))
            # add the train data
            #x_train_tmp = z = np.concatenate([total_density, rdf_max_position], axis=1)
            #x_train = np.r_[x_train, x_train_tmp]
            #y_train = np.r_[y_train, X]
            
            # calculate the fitness
            #p_fitness = FitnessFunction(total_density, AA_density, all_score)
            p_fitness = FitnessFunction(total_density, AA_density, max_RDF_core)
            #print(p_fitness)
            g_fitness = p_fitness.min()
            fitness_val_list.append(g_fitness)
            # get ech perfect postion
            gbest = pbest[p_fitness.argmin()] 
            print("Gbest particle num: %d" % int(p_fitness.argmin()+1))
            if str(g_fitness) == "nan":
                g_fitness = 10.000
            print("Gfitness value: %s" % str(g_fitness))
            print("Gbest value:")
            print(gbest)
            # save sigma and epsilon
            for i in range(particle_num):
                all_sigma_spsilon_file = open("MD_run/particle_%s/sigma_spsilon_%s.dat" % (str(i+1),str(i+1)),'a')
                all_sigma_spsilon_file.write("iter   sigma epsilon sigma epsilon ……\n")
                all_sigma_spsilon_file.write("%5d" % int(k+1))
                for num in range(MDparameter_num):
                    all_sigma_spsilon_file.write(" %8.3f" % X[i][num])
                all_sigma_spsilon_file.write("\n")
                all_sigma_spsilon_file.close()
            # Linearly Decreasing Weight, LDW  
            # # S hi Y, Eberhart R C .A modified particle sw arm optimizer
              #[ C] // Proceedings of the IEE E In t Conf on Evolu tionary
              #C om putation .An chorage:IEEE Press, 1998 :69-73
            w = (w_init-w_end)*(iter_num-(k+1))/iter_num + w_end
            # update the position and velocity of particle
            V = UpdateVelocity(V, X, pbest, gbest, c1, c2, w, max_vel)
            X = UpdatePosition(X, V)
        ##
        else:               
            # update the sigma and epsilon in the itp file
            for i in range(particle_num):
                itp_file = open("MD_run/particle_%s/atomtypes.itp" % str(i+1), 'w+')
                itp_file.write("[ atomtypes ]\n \
                                ;name   bond_type     mass     charge   ptype   sigma         epsilon       Amb\n")
                jj = 0
                while jj < len(CG_atom_type):
                    itp_file.write("%s      %s      0.00000  0.00000   A   %8.5e      %8.5e \n"  \
                                    % (CG_atom_type[jj], CG_atom_type[jj], X[i][2*jj], X[i][2*jj+1]))
                    jj += 1
                itp_file.close()    
                   
            # run MD simulation
            if type_system == "local":
                RunAllMDLocal(process_num, particle_num)         
            elif type_system == "slurm":
                RunAllMDSlurm(process_num, particle_num)
            else:
                print("!!!!!!!!!!!!!type_system error!!!!!!!!!!!!")
                sys.exit()
                    
            # get density
            total_density = []
            for i in range(particle_num):
                density = GetDensity(i)
                if str(density) == "-nan" or str(density) == "":
                    density = 1000
                total_density.append([float(density)/1000.0])
            total_density = np.array(total_density)
            # get score value
            all_score, rdf_max_position = GetAllscore(particle_num, atom_type_rdf_AA, CG_rdf_pair)
            max_RDF_core = GetMaxRDFcore(particle_num, AA_rdf_max_position, rdf_max_position)
            #print("Best score: %f" % float(np.min(all_score)))
            #print("Best score particle num: %d" % int(np.argmin(all_score)+1))
            print("Best score: %f" % float(np.min(max_RDF_core)))
            print("Best score particle num: %d" % int(np.argmin(max_RDF_core)+1))
            # add the train data
            if k >= collect_sets:
                x_train_tmp = z = np.concatenate([total_density, rdf_max_position], axis=1)
                x_train = np.r_[x_train, x_train_tmp]
                y_train = np.r_[y_train, X]
            
            # calculate the fitness
            #p_fitness2 = FitnessFunction(total_density, AA_density, all_score)
            p_fitness2 = FitnessFunction(total_density, AA_density, max_RDF_core)
            g_fitness2 = p_fitness2.min()
            #print(p_fitness2)
            # update the fitness for ech particle
            for j in range(particle_num):
                if  p_fitness2[j] < p_fitness[j]:
                    pbest[j] = X[j]
                    p_fitness[j] = p_fitness2[j]
                if  g_fitness2 < g_fitness:
                    gbest = X[p_fitness2.argmin()]
                    g_fitness = g_fitness2

            # save the fitness values
            fitness_val_list.append(g_fitness)
            print("Gbest particle num: %d" % int(p_fitness.argmin()+1))
            if str(g_fitness) == "nan":
                g_fitness = 10.000
            print("Gfitness value: %s" % str(g_fitness))           
            print("Gbest value:")
            print(gbest)

            # chaostic particle swarm optimization
            if k >= chaostic_begin:
                print("***** Add CPSO ******")
                # gbest to [0, 1]
                z0_tmp = []
                for jc in range(MDparameter_num):
                    if jc % 2 == 0:
                        # transformation
                        X_tmp = (gbest[jc]-sigma_min_max[0])/(sigma_min_max[1]-sigma_min_max[0])
                    else:
                        # transformation
                        X_tmp = (gbest[jc]-epsilon_min_max[0])/(epsilon_min_max[1]-epsilon_min_max[0])
                    z0_tmp.append(X_tmp)
                z0_tmp = np.array([z0_tmp])
                # Logistic function
                z = np.empty(shape=(0, MDparameter_num))
                for i in range(particle_num):
                    z = np.r_[z, z0_tmp]   
                    z0_tmp = np.multiply(4*z0_tmp, 1-z0_tmp)     
                # convert to real sigma and epsilon
                X_chaostic = []
                for i in range(particle_num):
                    tmp = []
                    for j in range(MDparameter_num):
                        if j % 2 == 0:
                            # transformation
                            X_tmp = sigma_min_max[0] + (sigma_min_max[1]-sigma_min_max[0])*z[i][j]
                        else:
                            # transformation
                            X_tmp = epsilon_min_max[0] + (epsilon_min_max[1]-epsilon_min_max[0])*z[i][j]
                        tmp.append(X_tmp)
                    X_chaostic.append(tmp)
                X_chaostic = np.array(X)
                print("chaostic sigma epsilon:")
                print(X_chaostic)
                # update the sigma and epsilon in the itp file
                for i in range(particle_num):
                    itp_file = open("MD_run/particle_%s/atomtypes.itp" % str(i+1), 'w+')
                    itp_file.write("[ atomtypes ]\n \
                                    ;name   bond_type     mass     charge   ptype   sigma         epsilon       Amb\n")
                    jj = 0
                    while jj < len(CG_atom_type):
                        itp_file.write("%s      %s      0.00000  0.00000   A   %8.5e      %8.5e \n"  \
                                        % (CG_atom_type[jj], CG_atom_type[jj], X_chaostic[i][2*jj], X_chaostic[i][2*jj+1]))
                        jj += 1
                    itp_file.close() 
                # run MD                   
                if type_system == "local":
                    RunAllMDLocal(process_num, particle_num)         
                elif type_system == "slurm":
                    RunAllMDSlurm(process_num, particle_num)
                else:
                    print("!!!!!!!!!!!!!type_system error!!!!!!!!!!!!")
                    sys.exit()
                # get density
                total_density = []
                for i in range(particle_num):
                    density = GetDensity(i)
                    if str(density) == "-nan" or str(density) == "":
                        density = 1000
                    total_density.append([float(density)/1000.0])
                total_density = np.array(total_density)
                # get score value
                all_score, rdf_max_position = GetAllscore(particle_num, atom_type_rdf_AA, CG_rdf_pair)
                max_RDF_core = GetMaxRDFcore(particle_num, AA_rdf_max_position, rdf_max_position)
                #print("Best score: %f" % float(np.min(all_score)))
                #print("Best score particle num: %d" % int(np.argmin(all_score)+1))
                print("Best score: %f" % float(np.min(max_RDF_core)))
                print("Best score particle num: %d" % int(np.argmin(max_RDF_core)+1))
                # add the train data
                x_train_tmp = z = np.concatenate([total_density, rdf_max_position], axis=1)
                x_train = np.r_[x_train, x_train_tmp]
                y_train = np.r_[y_train, X_chaostic]
            
                # calculate the fitness of CPSO
                #p_fitness_chaostic = FitnessFunction(total_density, AA_density, all_score)
                p_fitness_chaostic = FitnessFunction(total_density, AA_density, max_RDF_core)
                g_fitness_chaostic = p_fitness_chaostic.min()
                gbest_chaostic = X_chaostic[p_fitness_chaostic.argmin()]
                print("chaostic Gbest particle num: %d" % int(p_fitness_chaostic.argmin()+1))
                print("chaostic Gfitness value: %s" % str(g_fitness_chaostic))           
                print("chaostic Gbest value:")
                print(gbest_chaostic)
                # randomly replace the origin position X using gbest_chaostic
                tmp = random.randint(0, particle_num-2) # the last data is ANN model, can not replace
                X[tmp] = gbest_chaostic
                # end CPSO
                print("******** CPSO end *********")

            # save sigma and epsilon
            for i in range(particle_num):
                all_sigma_spsilon_file = open("MD_run/particle_%s/sigma_spsilon_%s.dat" % (str(i+1),str(i+1)),'a')
                all_sigma_spsilon_file.write("%5d" % int(k+1))
                for num in range(MDparameter_num):
                    all_sigma_spsilon_file.write(" %8.3f" % X[i][num])
                all_sigma_spsilon_file.write("\n")
                all_sigma_spsilon_file.close()
            # Linearly Decreasing Weight, LDW  
            # # S hi Y, Eberhart R C .A modified particle sw arm optimizer
              #[ C] // Proceedings of the IEE E In t Conf on Evolu tionary
              #C om putation .An chorage:IEEE Press, 1998 :69-73
            w = (w_init-w_end)*(iter_num-(k+1))/iter_num + w_end
            # update the position and velocity of particle
            V = UpdateVelocity(V, X, pbest, gbest, c1, c2, w, max_vel)
            X = UpdatePosition(X, V)
            
            # add ANN model
            if  k >= ANN_begin:
                print("***** Add ANN ******")
                predictions = ANN(x_train, y_train, x_predict, BATCH_SIZE, EPOCHS)
                print("ANN predictions value:")
                print(predictions)
                if predictions[0][0] != 0 and predictions[0][1] != 0:
                    X[particle_num-1] = predictions
                    print("******** ANN end (added) *********")
                else:
                    print("******** ANN end (not added) *********")
        #
        gbest_file.write("%5d" % int(k+1))
        for num in range(MDparameter_num):
            gbest_file.write("%8.3f" % gbest[num])
        gbest_file.write("\n")
        
    # print optimal solution
    fitness_val_file = open("fitness_value.dat",'w+')
    fitness_val_file.write("iter   fitness_value\n")
    for ii in range(iter_num):
        fitness_val_file.write("%5d%20.10f\n" % (int(ii+1), fitness_val_list[ii]))
        
    fitness_val_file.close()
    gbest_file.close()    
    print("----------------------------------------------------------------------")
    print("The optimal value of fitness is: %.5f" % fitness_val_list[-1])
    print("The optimal value of sigma and epsilon are:")
    print(gbest)
    print("----------------------------------------------------------------------")
    # save ANN data sets
    train_file = open("train.dat",'w+')
    for info in range(len(x_train)):
        train_file.write("x: ")
        for info1 in range(len(x_train[info])):
            train_file.write(" %f," % x_train[info][info1])
        train_file.write(" y: ")
        for info1 in range(len(y_train[info])):
            train_file.write(" %f," % y_train[info][info1])
        train_file.write("\n")
    train_file.close()

if __name__ == '__main__':
    ##################################################################################################################
    # PSO algorithm setup
    particle_num = 4
    iter_num = 6
    chaostic_begin = 3 # iteration of add the CPSO
    max_vel = 0.1
    # ANN setup
    EPOCHS = 2000
    BATCH_SIZE = 32
    collect_sets = 1 # iteration of collect train data, collect_sets >= ANN_begin
    ANN_begin = 3 # iteration of add the ANN model
    # MD setup
    begin_time = 100  # ps
    process_num = 4  # parallel number of runing MD
    MD_file_path = "MDfile"  # contain: mdp file, CG gro file, CG top/itp file, run.sh
    CG_atom_type = ["OW_spc"] # all atom type
    CG_rdf_pair = [["OW_spc","OW_spc"]]
    sigma_min_max = [0.1, 0.4]
    epsilon_min_max = [3, 5]
    # type of running system
    type_system = "local"  # local or slurm
    # AA simulation density
    AA_density = 0.974787 # g/cm^3
    AA_rdf_path = "AA_MD" # the rdf name must be: CGtype_CGtype.xvg, for example: OW_spc_OW_spc.xvg, OW_spc is CG type
    ##################################################################################################################
    #
    assert(ANN_begin>=collect_sets)
    MDparameter_num = len(CG_atom_type) * 2
    # create the MD folder
    os.system("rm -rf MD_run")
    os.system("mkdir MD_run")
    # create the index file
    os.system('echo "del 0-20" > index')
    for CGtype in CG_rdf_pair:
        os.system('echo "t %s" >> index' % str(CGtype[0]))
        os.system('echo "t %s" >> index' % str(CGtype[1]))
    os.system('echo "q" >> index')
    # create runing GMX file
    if type_system == "local":
        os.system('echo "#!/usr/bin/env bash" > run.sh')
        os.system('echo "rm ./#* *.xvg *.edr *.xtc *.trr *.tpr" >> run.sh')
        os.system('echo "gmx grompp -f minim.mdp -c init.gro -p top.top -o em.tpr -maxwarn 5" >> run.sh')
        os.system('echo "gmx mdrun -v -deffnm em -nt 6" >> run.sh')
        os.system('echo "gmx grompp -f md.mdp -c em.gro -p top.top -o md.tpr -maxwarn 5" >> run.sh')
        os.system('echo "gmx mdrun -v -deffnm md -nt 6" >> run.sh')
        os.system('echo "gmx energy -f md.edr -o den.xvg -b %s <<< Density >& den.log" >> run.sh' % begin_time)    
        os.system('echo "gmx make_ndx -f md.tpr < index" >> run.sh')
        num = 0
        for CGtype in CG_rdf_pair:
            os.system('echo "echo -e %s \' \\\RRR %s \' | gmx rdf -f md.xtc -s md.tpr -n index.ndx -b %s -o %s_%s.xvg"  >> run.sh' %
                    (num, num+1, begin_time, CGtype[0], CGtype[1]))
            num += 2
        os.system('echo "echo done > tag_finished" >> run.sh')
        os.system('sed -i "s/RRR/n/g" run.sh')
    elif type_system == "slurm":
        os.system('echo "#!/usr/bin/env bash" > run.sh')
        os.system('echo "#SBATCH -J gromacs" >> run.sh')
        os.system('echo "#SBATCH -n 32" >> run.sh')
        os.system('echo "#SBATCH -N 1" >> run.sh')
        os.system('echo "#SBATCH -o out.out" >> run.sh')
        os.system('echo "#SBATCH -e out.err" >> run.sh')
        os.system('echo "#SBATCH -p wzhdnormal" >> run.sh')
        os.system('echo "total_core=32" >> run.sh')
        
        os.system('echo "module use  /public/software/modules" >> run.sh')
        os.system('echo "module load mpi/hpcx/gcc-7.3.1" >> run.sh')
        os.system('echo "module load compiler/intel/2017.5.239" >> run.sh')
        os.system('echo "module load apps/gromacs/2019.5/hpcx-intel2017" >> run.sh')
        os.system('echo "rm ./#* *.xvg *.edr *.xtc *.trr *.tpr" >> run.sh')
        os.system('echo "gmx_mpi grompp -f minim.mdp -c init.gro -p top.top -o em.tpr -maxwarn 5" >> run.sh')
        os.system('echo "mpirun -np \$total_core gmx_mpi mdrun -v -deffnm em" >> run.sh')
        os.system('echo "gmx_mpi grompp -f md.mdp -c em.gro -p top.top -o md.tpr -maxwarn 5" >> run.sh')
        os.system('echo "mpirun -np \$total_core gmx_mpi mdrun -v -deffnm md" >> run.sh')
        os.system('echo "gmx_mpi energy -f md.edr -o den.xvg -b %s <<< Density >& den.log" >> run.sh' % begin_time)    
        os.system('echo "gmx_mpi make_ndx -f md.tpr < index" >> run.sh')
        num = 0
        for CGtype in CG_rdf_pair:
            os.system('echo "echo -e %s \' \\\RRR %s \' | gmx_mpi rdf -f md.xtc -s md.tpr -n index.ndx -b %s -o %s_%s.xvg"  >> run.sh' %
                    (num, num+1, begin_time, CGtype[0], CGtype[1]))
            num += 2
        os.system('echo "echo done > tag_finished" >> run.sh')
        os.system('sed -i "s/RRR/n/g" run.sh')
    else:
        print("!!!!!!!!!!!!!type_system error!!!!!!!!!!!!")
        sys.exit()

    # copy file
    for i in range(particle_num):
        os.system("mkdir MD_run/particle_%s" % str(i+1))
        os.system("cp MDfile/* MD_run/particle_%s" % str(i+1))
        os.system("cp index MD_run/particle_%s" % str(i+1))
        os.system("cp run.sh MD_run/particle_%s" % str(i+1))
    ##
    main(MDparameter_num, particle_num, iter_num, max_vel, CG_atom_type, AA_density, 
         process_num, sigma_min_max, epsilon_min_max, chaostic_begin, AA_rdf_path, 
         CG_rdf_pair, ANN_begin, EPOCHS, BATCH_SIZE, collect_sets, type_system)