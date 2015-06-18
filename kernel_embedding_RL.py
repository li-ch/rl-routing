import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.linalg import det
from scipy.linalg import cholesky
from math import pi
import os
import time
from pylab import * 

class KEMDRL:
    def train(self, data_path, action_name, reward_name, feature_name_curr, feature_name_next, feature_type, feature_weight=[], start=0, length=-1, model_path='KEMD_model.txt', max_iter=50, lda=1e-1, learning_rate=0.01):
        feature_name_curr = feature_name_curr.split(':')
        feature_name_next = feature_name_next.split(':')
        feature_type = feature_type.split(':')
        feature_weight = feature_weight.split(':')
        num_of_feature = len(feature_name_curr)
        cols_curr = feature_name_curr[:]
        cols_next = feature_name_next[:]
        
        df = pd.read_csv(data_path)
        data_len = len(df)
 
        if length == -1:
           length = data_len
        
        if feature_weight == []:
           feature_weight = [1.0/num_of_feature] * num_of_feature
        
       
        
        Len_rr = length/3
        idx_sr = np.random.permutation(length)
        idx_sr = idx_sr[0:Len_rr]
        
        S_t = df.loc[idx_sr, cols_curr]
        S_t = np.array(S_t).astype(float)
        S_t = S_t.reshape((Len_rr,num_of_feature))
        
        A_t = df.loc[idx_sr, action_name]
        A_t = np.array(A_t).astype(float)
        A_t = A_t.reshape((Len_rr,1))

        S = df.loc[start:start+length-1, cols_curr]
        S = np.array(S).astype(float)
       
        S = S.reshape((length,num_of_feature))
        
        S_p = df.loc[start:start+length-1, cols_next]
        S_p = np.array(S_p).astype(float)
        S_p = S_p.reshape((length,num_of_feature))

        A = df.loc[start:start+length-1, action_name] 
        A = np.array(A).astype(float)
        A = A.reshape((length,1)) 
        
        R = df.loc[start:start+length-1, reward_name]
        R = np.array(R).astype(float)
        R = R.reshape((length, 1))
        
        alpha_init = np.ones((Len_rr, 1), dtype = float)/Len_rr 
        yita_init = 0.5 
        ep = 0.2 
      
        logtheta = np.log(1)  
        logeta = np.log(0.5)        
        

        yita,alpha,Yita,Alpha,G,Beta,K_tp,K_ts, K_tt = self.kernel_embedding_optimize_g_RL(S_t,S_p,S,A,R,alpha_init,yita_init,lda,logtheta,logeta,ep,feature_type,max_iter, learning_rate)        
        
        figure
        subplot(2,1,1)
        plot(Alpha.transpose()) 
        subplot(2,1,2)
        plot(Yita)
               
        
        inv_L, Q, Pi, H, Lambda, Gamma, X, Y, M , C = self.kernel_embedding_model(yita, alpha, Beta, K_tp, K_ts, K_tt, A, R, logtheta, logeta, Len_rr, lda)    
        
        model = KEMDModel()
        model.setActionName(action_name)
        model.setRewardName(reward_name)
        model.setCurrFeatureName(feature_name_curr)
        model.setNextFeatureName(feature_name_next)
        model.setLogtheta(logtheta)
        model.setLogdelta(logeta)
        model.setFeatureType(feature_type)
        model.setFeatureWeight(feature_weight)
        model.setMaxIter(max_iter)
        model.setLda(lda)
        model.setLearningRate(learning_rate)
        model.setIdxSrSt(S_t)
        model.setIdxSrAt(A_t)
        model.setInvL(inv_L)
        model.setAlpha(alpha)
        model.setYita(yita)
        model.setPi(Pi)
        model.setH(H)
        model.setGamma(Gamma)
        model.setX(X)
        model.setY(Y)
        model.setM(M)
        model.setC(C)
        model.save(model_path)
        return model
        
    def kernel_embedding_optimize_g_RL(self,S_t, S_p, S, A, R, alpha_init, yita_init, lda, logtheta, logeta, ep, feature_type, MaxIter=100, learning_r=0.005):
        
        yita = yita_init
        alpha = alpha_init
        Yita = []
        Yita.append(np.array(yita))
    
    
        G = []
	
   
        stepsize = 0.001
        thrshold = 0.00001
	
        Alpha = np.zeros((len(alpha_init),MaxIter))
        for j in range(0,len(alpha_init)):
            Alpha[j,0] = alpha_init[j,0]

	
        K_ss = KEMDOPERATION.kernel_embedding_D(S, S, feature_type)
        K_ss = KEMDOPERATION.kernel_embedding_K(K_ss, np.exp(logtheta), [np.exp(logeta)] )
	
        K_aa = KEMDOPERATION.kernel_embedding_D(A, A, ['link'])
        K_aa = KEMDOPERATION.kernel_embedding_K(K_aa, np.exp(logtheta), [np.exp(logeta)])
	
        K_sa = K_ss*K_aa
	
        K_ts = KEMDOPERATION.kernel_embedding_D(S_t, S, feature_type)
        K_ts = KEMDOPERATION.kernel_embedding_K(K_ts, np.exp(logtheta), [np.exp(logeta)])   
	    
        K_tt = KEMDOPERATION.kernel_embedding_D(S_t, S_t, feature_type)
        K_tt = KEMDOPERATION.kernel_embedding_K(K_tt, np.exp(logtheta), [np.exp(logeta)])
    
        n = len(S)
        inv_K = inv(K_sa + lda * np.identity(n))
	
        Beta = np.dot(inv_K, K_sa)

        K_tp = KEMDOPERATION.kernel_embedding_D(S_t, S_p, feature_type)
        K_tp = KEMDOPERATION.kernel_embedding_K(K_tp, np.exp(logtheta), [np.exp(logeta)])
    
        print "Minimizing G function:"  
	
        for i in range(0,MaxIter):
          yita_prev = yita 
          alpha_prev = alpha
          Yita.append(yita)
        
          for j in range(0,len(alpha_init)):
            Alpha[j,i] = alpha[j,0]


          g_prev = self.kernel_embedding_g_2(n, K_tp, Beta, K_ts, R, alpha_prev, yita_prev, lda, ep)
          print g_prev
	    
          G.append(g_prev)
          g_new = self.kernel_embedding_g_2(n, K_tp, Beta, K_ts, R, alpha_prev, yita+stepsize, lda, ep)
          yita = yita - learning_r*(g_new - g_prev)/stepsize
		
	    #print alpha[0]	
		
		
          for k in range(0,len(alpha)):
            alpha_stepsize = alpha_prev
            alpha_stepsize[k] = alpha_stepsize[k] + stepsize
            g_new = self.kernel_embedding_g_2(n, K_tp, Beta, K_ts, R, alpha_stepsize, yita_prev, lda, ep)
                
            alpha[k] = alpha[k]-learning_r*(g_new-g_prev)/stepsize
             
          if abs(yita-yita_prev)<thrshold and max(abs(alpha-alpha_prev))<thrshold:
           break
         
         
        return yita,alpha,Yita,Alpha,G,Beta,K_tp,K_ts, K_tt

    def kernel_embedding_model(self, yita, alpha, Beta, K_tp, K_ts, K_tt, A, R, logtheta, logeta, Len_sr, lda):

        I_sr = np.identity(Len_sr, dtype = float)
    
        L = cholesky((K_tt + lda * I_sr))
    
        inv_L = inv(L).transpose()
        Q = np.dot(np.transpose(K_ts), inv_L)
    
        Pi = inv( np.dot(np.transpose(Q), Q) + lda * I_sr)
    
        H = np.dot( K_tp, Q)
    
        BLK = np.dot(K_tp,Beta)-K_ts
        Delta = R + np.dot(np.transpose(BLK),alpha)
        W = np.exp(Delta/yita)
        C = sum(W[:,0])/len(W[:,0])
        
        W = W/C
        Lambda = np.diag(W[:,0]) 
        #Lambda = np.identity(len(W[:,0]),dtype = float)
    
        Gamma = np.dot( np.dot(K_ts, Lambda), np.transpose(K_ts))
    
        X = inv(Gamma + lda * K_tt + lda**2 * I_sr)
        Y = np.dot(K_ts, A)
    
        M = np.dot(X, Y)
    
        return inv_L, Q, Pi, H, Lambda, Gamma, X, Y, M, C 
    
    def kernel_embedding_g_2( self, n, K_tp, Beta, K_ts, R, alpha, yita, lda, ep):
    
        Delta = 0
        yita = np.exp(yita)

        for i in range(0,n):
           delta = R[i] + np.dot(np.transpose(alpha), np.dot(K_tp, Beta[:,i])- K_ts[:,i])
           Delta = Delta + np.exp(delta/yita)
           g = yita * ep + yita * np.log(Delta/n)
        return g
    
class KEMDModel: 
    
    def predictSingle(self, sample, sample_id, result_file = 'singleResult.txt'):
         #tic = time.time()
         test_data = np.array([[int(i) for i in sample.split(':')]])
         
         #simTalbe1 = self.getSimTable1()
         #simTable2 = self.getSimTable2()
         
         Len_sr = len(self.getSimTableSt()[0][0][:])
         
         K_star= np.ones((1,Len_sr), dtype = float)
         
         num_of_feature = len(self.getCurrFeatureName())
         
         
        
         for k in range(0, num_of_feature):
             K_star = K_star * (self.getSimTableSt()[k][test_data[0,k]][:])
             
         
         K_star = np.exp(self.getLogtheta()) * (K_star )
       
        
         
         f_mean = (np.dot(K_star, self.getM()))
         f_std = np.sqrt(np.exp(self.getLogtheta())+self.getLda()- np.dot(K_star, np.dot(self.getX(), K_star.transpose())))   

         a = f_std * np.random.randn(1) + f_mean         
         
       
         return a[0]
    
    def update_online(self, sample, sample_id):
    
        test_data = np.array([[int(i) for i in sample.split(':')]])
        Len_sr = len(self.getSimTableSt()[0][0][:])
                
        num_of_feature = len(self.getFeatureType())
        s = test_data[0,0:num_of_feature]
        
        s = s.reshape((1,num_of_feature))
        sp = test_data[0,num_of_feature:2*num_of_feature]
        sp = sp.reshape((1,num_of_feature))
        a = test_data[0,2*num_of_feature:2*num_of_feature+1]
        a = a.reshape((1,1))
        r = test_data[0,2*num_of_feature+1]
 
        K_ts= np.ones((1,Len_sr), dtype = float) 
        
        for k in range(0, num_of_feature):
           K_ts = K_ts * (self.getSimTableSt()[k][s[0,k]][:])
             
        K_ts = np.exp(self.getLogtheta()) * (K_ts )
        
        
        
        K_ta = np.exp(self.getLogtheta())*(self.getSimTableAt()[a[0,0]][:])
             
           
    
        K_sa = K_ts * K_ta
      
        K_sa = K_sa.transpose()
        K_ts = K_ts.transpose()
    
        K_tp= np.ones((1,Len_sr), dtype = float) 
        
        for k in range(0, num_of_feature):
           K_tp = K_tp * (self.getSimTableSt()[k][sp[0,k]][:])
             
        K_tp = np.exp(self.getLogtheta()) * (K_tp )
        K_tp =  K_tp.transpose()
    
     
        q = np.dot(self.getInvL(), K_sa)
    
        Pi_bk = self.getPi()
    
        dnm = (1+ np.dot(q.transpose(), np.dot(self.getPi(), q)))
        qq = np.dot(q, q.transpose())
    
        Pi = self.getPi() - 1.0/dnm * np.dot(self.getPi(), np.dot(qq, self.getPi()))
    
    
        H_bk = self.getH() 
       
        H = self.getH() + np.dot(K_tp, q.transpose())
    
       #ddelta = -1.0/lda * np.dot(alpha.transpose(), np.dot(H, np.dot(Pi, q))) + 1.0/lda * (1 - np.dot(q.transpose(), np.dot(Pi, q))) * np.dot(q.transpose(), K_tp)
        delta = r + 1.0/self.getLda() * np.dot(self.getAlpha().transpose(), np.dot(H_bk, np.dot(np.trace(qq)/dnm * Pi_bk - self.getPi(), q))) + self.getLda() * (1+ np.trace(qq))/dnm * np.dot(self.getAlpha().transpose(), K_tp) -np.dot(self.getAlpha().transpose(), K_ts)
 
   
        c = np.exp(delta/self.getYita())/self.getC()
        Gamma = self.getGamma() + c * np.dot(K_ts, K_ts.transpose())
    
        X_BLK = np.dot(K_ts, np.dot(K_ts.transpose(), self.getX()))
        X_nm = c * np.dot(self.getX(), X_BLK)
        X_dnm = 1 + c * np.trace(X_BLK)
        X = self.getX() - 1/X_dnm * X_nm 
    
        Y = self.getY() + a[0,0] * K_ts 
        M = np.dot(self.getX(), self.getY())
       
        self.setPi(Pi)
        self.setH(H)
        self.setGamma(Gamma)
        self.setX(X)
        self.setY(Y)
        self.setM(M)
    
        return Pi, H, Gamma, X, Y, M


        
    def load(self, model_path):
        f = open(model_path)
        f.readline()
        self.setActionName(f.readline().strip()) 
        
        f.readline()
        self.setRewardName(f.readline().strip().split())
        
        f.readline()
        self.setCurrFeatureName(f.readline().strip().split(','))
        
        f.readline()
        self.setNextFeatureName(f.readline().strip().split(','))

        f.readline()
        self.setFeatureType(f.readline().strip().split(','))
 
        f.readline();
        parts = f.readline().strip().split(',');
        self.setFeatureWeight([float(i) for i in parts])

        f.readline()
        self.setMaxIter(int(f.readline().strip())) 

        f.readline();
        self.setLda(float(f.readline().strip()))

        f.readline()
        self.setLearningRate(float(f.readline().strip()))
        
        Idx_sr_size = f.readline().strip().split(':')[-1].split(',')
        m = int(Idx_sr_size[0])
        n = int(Idx_sr_size[1])
        Idx_sr_array = np.array(f.readline().strip().split(','),dtype=np.float)
        self.setIdxSrSt(Idx_sr_array.reshape(m,n))
        
        Idx_sr_size = f.readline().strip().split(':')[-1].split(',')
        m = int(Idx_sr_size[0])
        n = int(Idx_sr_size[1])
        Idx_sr_array = np.array(f.readline().strip().split(','),dtype=np.float)
        self.setIdxSrAt(Idx_sr_array.reshape(m,n))
        
        invL_size = f.readline().strip().split(':')[-1].split(',')
        m = int(invL_size[0])
        n = int(invL_size[1])
        invL_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setInvL(invL_array.reshape(m,n))
                
        alpha_size = f.readline().strip().split(':')[-1].split(',')
        m = int(alpha_size[0])
        n = int(alpha_size[1])
        alpha_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setAlpha(alpha_array.reshape(m,n))
               
        f.readline()
        self.setYita(float(f.readline().strip()))
        
        pi_size = f.readline().strip().split(':')[-1].split(',')
        m = int(pi_size[0])
        n = int(pi_size[1])
        pi_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setPi(pi_array.reshape(m,n))
        
        h_size = f.readline().strip().split(':')[-1].split(',')
        m = int(h_size[0])
        n = int(h_size[1])
        h_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setH(h_array.reshape(m,n))
        
        gamma_size = f.readline().strip().split(':')[-1].split(',')
        m = int(gamma_size[0])
        n = int(gamma_size[1])
        gamma_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setGamma(gamma_array.reshape(m,n))
     
        x_size = f.readline().strip().split(':')[-1].split(',')
        m = int(x_size[0])
        n = int(x_size[1])
        x_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setX(x_array.reshape(m,n))     
        
        y_size = f.readline().strip().split(':')[-1].split(',')
        m = int(y_size[0])
        n = int(y_size[1])
        y_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setY(y_array.reshape(m,n))
        
        m_size = f.readline().strip().split(':')[-1].split(',')
        m = int(m_size[0])
        n = int(m_size[1])
        m_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setM(m_array.reshape(m,n))       
        
        f.readline()
        self.setC(float(f.readline().strip()))
        
        f.readline()
        self.setLogtheta(float(f.readline().strip()))  

        f.readline()
        self.setLogdelta(float(f.readline().strip()))          
        
       
        feature_type = self.getFeatureType()
        num_of_feature = len(feature_type)
        
        simTableSt = []
       
        
        #theta1 = self.getTheta1()
        delta = np.exp(self.getLogdelta())
        
        #theta2 = self.getTheta2()
      
        
        Idx_srst = self.getIdxSrSt()
        
        for k in range(0,num_of_feature):
            Type = feature_type[k]
            data_sr = Idx_srst[:,k]
            data_sr = data_sr.reshape((len(data_sr),1))
            
            if Type == 'IP':
                data = np.array([i for i in range(0,256)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
            elif Type == 'Port': 
                data = np.array([i for i in range(0, 65536)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
            elif Type == 'Categorical':
                data = np.array([i for i in range(0,2)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
                
            elif Type == 'Hour':
                data = np.array([i for i in range(0,24)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
            elif Type == 'Minute':
                data = np.array([i for i in range(0,60)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
            elif Type == 'ActiveFlow': 
                data = np.array([float(i) for i in range(0, 100)])
                data = np.transpose(data) 
                data = data.reshape((len(data), 1))
            elif Type == 'Link':
                 data = np.array([i for i in range(0, 100)])  # assume 100 = # of edge switches * # of core switches
            else:
                print "Error"
            
            D_k = KEMDOPERATION.kernel_embedding_D(data, data_sr, [Type])
            
            
            K_k = KEMDOPERATION.kernel_embedding_K(D_k, 1, delta )
            
            
            simTableSt.append(K_k)
        #print delta
        
        self.setSimTableSt(simTableSt)
        
        idx_srat = self.getIdxSrAt()
        data = np.array([i for i in range(0, 100)])  # 100 is the number of possible paths
        data = np.transpose(data) 
        data = data.reshape((len(data), 1))
        data_sr = idx_srat.reshape((len(idx_srat),1))
        D = KEMDOPERATION.kernel_embedding_D(data, data_sr, ['Link'])
        K = KEMDOPERATION.kernel_embedding_K(D, 1, delta)
        
        self.setSimTableAt(K)
        
        f.close();
        
    def save(self, model_path): 
        f = open(model_path, 'wb')
        f.write('## ACTION_NAME:1,1\n')
        f.write(self.getActionName() + '\n')
        f.write('## REWARD_NAME: 1,1 \n')
        f.write(self.getRewardName() + '\n')        
        f.write('## CURR_FEATURE_NAME:1,')
        f.write(str(len(self.getCurrFeatureName())) + '\n')
        f.write(','.join(self.getCurrFeatureName()) + '\n')
        f.write('## NEXT_FEATURE_NAME:1,')
        f.write(str(len(self.getNextFeatureName())) + '\n')
        f.write(','.join(self.getNextFeatureName()) + '\n')    
        f.write('##FEATURE_TYPE:1,')
        f.write(str(len(self.getFeatureType())) + '\n')
        f.write(','.join(self.getFeatureType()) + '\n')
        f.write('##FEATURE_WEIGHT:1,')
        f.write(str(len(self.getFeatureWeight())) + '\n')
        f.write(','.join([str(i) for i in self.getFeatureWeight()]) + '\n')
        f.write('## MAX_ITERATIONS: 1, 1\n')
        f.write(str(self.getMaxIter())+'\n')
        f.write('## LDA: 1, 1\n')
        f.write(str(self.getLda()) + '\n')
        f.write('## LEARNING_RATE: 1,1\n')
        f.write(str(self.getLearningRate()) + '\n')
        idxSrSt = self.getIdxSrSt()
        (m, n) = idxSrSt.shape
        f.write('##IDXSRST:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(idxSrSt[a][b])
                l += ','
        f.write(l[:-1] + '\n')
        
        idxSrAt = self.getIdxSrAt()
        (m, n) = idxSrAt.shape
        f.write('##IDXSRAT:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(idxSrAt[a][b])
                l += ','
        f.write(l[:-1] + '\n') 
        
        invL = self.getInvL()
        (m, n) = invL.shape
        f.write('## INV_L:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(invL[a][b]))
                l += ', '
        f.write(l[:-2] + '\n')
        
        
        
        alpha = self.getAlpha()
        (m, n) = alpha.shape
        f.write('##ALPHA:' + str(m) + ',' +str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(alpha[a][b])
                l += ','
        f.write(l[:-1] + '\n')
        
        f.write('##YITA:1,1\n')
        f.write(str(self.getYita()[0]) + '\n')
     
        Pi = self.getPi()
        (m, n) = Pi.shape
        f.write('## Pi:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(Pi[a][b]))
                l += ', '
        f.write(l[:-2] + '\n') 

        H = self.getH()
        (m, n) = H.shape
        f.write('## Pi:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(H[a][b]))
                l += ', '
        f.write(l[:-2] + '\n') 

        Gamma = self.getGamma()
        (m, n) = Gamma.shape
        f.write('## Pi:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(Gamma[a][b]))
                l += ', '
        f.write(l[:-2] + '\n') 

        X = self.getX()
        (m, n) = X.shape
        f.write('## X:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(X[a][b]))
                l += ', '
        f.write(l[:-2] + '\n')   

        Y = self.getY()
        (m, n) = Y.shape
        f.write('## Y:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(Y[a][b]))
                l += ', '
        f.write(l[:-2] + '\n')   

        M = self.getM()
        (m, n) = M.shape
        f.write('## M:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(M[a][b]))
                l += ', '
        f.write(l[:-2] + '\n')         
        
        f.write('## C: 1,1\n')
        f.write(str(self.getC()) + '\n')
        
        f.write('## LOGTHETA: 1,1\n')
        f.write(str(self.getLogtheta()) + '\n')
        
        f.write('## LOGDELTA: 1,1\n')
        f.write(str(self.getLogdelta()) + '\n')
        
    def setActionName(self, name):
        self.ActionName = name
        
    def setRewardName(self, name):
        self.RewardName = name
        
    def setCurrFeatureName(self, name):
        self.CurrFeatureName = name 
        
    def setNextFeatureName(self, name):
        self.NextFeatureName = name 

    def setFeatureType(self, feature_type):
        self.FeatureType = feature_type
        
    def setFeatureWeight(self,weight):
        self.featureWeight = weight
        
    def setMaxIter(self,iter):
        self.maxIter = iter
        
    def setLda(self,r):
        self.lda = r
        
    def setLearningRate(self,rate):
        self.learningRate = rate
     
    def setIdxSrSt(self,St):
        self.idxSrSt = St
   
    def setIdxSrAt(self, At):
        self.idxSrAt = At
        
    def setInvL(self, invL):
        self.InvL = invL

    def setAlpha(self, alpha):
        self.Alpha = alpha

    def setYita(self, yita):
        self.yita = yita
    
    def setPi(self, Pi):
        self.Pi = Pi
     
    def setH(self, H):
        self.H = H 
       
    def setGamma(self, Gamma):
        self.Gamma = Gamma
        
    def setX(self,X):
        self.X = X 
     
    def setY(self, Y):
        self.Y = Y 

    def setM(self, M):
        self.M = M 
        
    def setC(self,C):
        self.C= C 
            
    def setSimTableSt(self,simTableSt):  
        self.simTableSt = simTableSt 
    
    def setSimTableAt(self, simTableAt):
        self.simTableAt = simTableAt 
        
    def setLogtheta(self, logtheta):
        self.logtheta = logtheta 

    def setLogdelta(self,logdelta):
        self.logdelta = logdelta 
        

    def getActionName(self):
        return self.ActionName
        
    def getRewardName(self):
        return self.RewardName
        
    def getCurrFeatureName(self):
        return self.CurrFeatureName 
        
    def getNextFeatureName(self):
        return self.NextFeatureName  

    def getFeatureType(self):
        return self.FeatureType
        
    def getFeatureWeight(self):
        return self.featureWeight
        
    def getMaxIter(self):
        return self.maxIter
        
    def getLda(self):
        return self.lda 
        
    def getLearningRate(self):
        return self.learningRate 
     
    def getIdxSrSt(self):
        return self.idxSrSt
   
    def getIdxSrAt(self):
        return self.idxSrAt 
        
    def getInvL(self):
        return self.InvL 

    def getAlpha(self):
        return self.Alpha

    def getYita(self):
        return self.yita 
    
    def getPi(self):
        return self.Pi 
     
    def getH(self):
        return self.H  
       
    def getGamma(self):
        return self.Gamma 
        
    def getX(self):
        return self.X 
     
    def getY(self):
        return self.Y  

    def getM(self):
        return self.M 
        
    def getC(self):
        return self.C 
            
    def getSimTableSt(self):  
        return self.simTableSt 
        
    def getSimTableAt(self):  
        return self.simTableAt 
        
    def getLogtheta(self):
        return self.logtheta 

    def getLogdelta(self):
        return self.logdelta 
        
        
class KEMDOPERATION:
    @staticmethod
    def delta_port(x):
        if x >= 0 and x <= 1023:
           delta=0
        elif x >= 1024 and x <= 49151:
           delta=1
        else: 
           delta=2
        return delta
        
    @staticmethod
    def dist_port(x1,x2):
        delta_x1 =  GaussOperation.delta_port(x1)
        delta_x2 = GaussOperation.delta_port(x2)
        if x1 == x2:
           dist = float(0)
        elif delta_x1 == delta_x2:
           dist = float(1)
        elif (delta_x1 == 0 or delta_x1 ==1) and (delta_x2 == 0 or delta_x2 == 1):
           dist = float(2) 
        else:
            dist = float (4) 
        return dist
  
    @staticmethod
    def convert_to_bit(ip):
        ret = ''
        v = int (ip)
        while v > 0:
           ret += str(v % 2)
           v = v / 2
        if len(ret) < 8 :
           ret += '0' * (8 - len(ret))
 
        return ret[::-1]

    @staticmethod
    def is_binary(num):
        b= str(num)
        for  i in b:
            if i == '0' or i == '1':
               continue
            return False
        if len(b) == 8:
            return True
        else:
            return False

    @staticmethod
    def dist_IP(ip1, ip2):
        L = 0
        IP1 = int (ip1)
        IP2 = int (ip2)
        if not GaussOperation.is_binary(IP1):
           IP1  = GaussOperation.convert_to_bit(IP1)
        
        if not GaussOperation.is_binary(IP2):
           IP2 = GaussOperation.convert_to_bit(IP2)

        Len = min(len(IP1), len(IP2))
  
        for k in range(0,Len):
            if IP1[k] != IP2[k]:
               break
            L = L+1
        d = np.log(float(Len+1)/float(L+1))

        return d

    @staticmethod 
    def dist_protocol(x1,x2):
        if x1== x2:
           dist = 0
        else: 
           dist = 1
        return dist
    
    @staticmethod
    def kernel_embedding_K(dist,theta,delta):
        Len = len(dist)    
       
        m,n = dist[0].shape

   
        y = np.ones((m,n), dtype = float)
        
        for i in range(0, Len):
            y = y * np.exp(-dist[i]/2/delta)  #[i]

        y = theta * (y )
        return y

    @staticmethod
    def kernel_embedding_D(data, data_sr, feature_type):
        
        len1 = len(data)
        len2 = len(data_sr)

        xx1 = np.transpose(data)
        xx2 = np.transpose(data_sr)

        temp = []
        for x in xx1: 
            temp.append(x.tolist()) 
        xx1 = temp 
    
        temp = []
        for x in xx2: 
            temp.append(x.tolist()) 
        xx2 = temp
   
        num_of_feature = len(feature_type)
        K = []
        for i in range(0, num_of_feature):
            K_k = np.zeros((len1, len2), dtype = float)
            K.append(K_k)
        dist_x1_x2 = 0.0

        for i in range(0, len1):
            for j in range(0,len2):
                for k in range(0, num_of_feature):
                    Type = feature_type[k]
                    x1 = xx1[k]
                    x2 = xx2[k]
                    if Type == 'numeric':
                       dist_x1_x2 = (x1[i]- x2[j])**2/np.abs(x1[i]*x2[j])
                    elif Type == 'IP':
                       dist_x1_x2 = (GaussOperation.dist_IP(x1[i],x2[j]))**2
                    elif Type == 'Port':
                        dist_x1_x2 = (GaussOperation.dist_port(x1[i],x2[j]))**2
                    elif Type == 'Categorical':
                        dist_x1_x2 = (GaussOperation.dist_protocol(x1[i],x2[j]))**2
                    elif Type == 'ActiveFlow':
                        dist_x1_x2 = float((x1[i] - x2[j]) **2 )
                    elif Type == 'Link':
                        if x1[i]==x2[j]:
                           dist_x1_x2 = 0.0
                        else: 
                           dist_x1_x2 = 1.0 
                    else: 
                        dist_x1_x2 = 0.0

                    K[k][i][j] = dist_x1_x2

        return K
        
    
         
         
if __name__ == '__main__':
    input = 'C:\\Users\\User\\Desktop\\RL_KEMD\\KEMD.csv'
    #input = 'C:\\Users\\User\\Desktop\\Flow Size Estimation Code and Specification\\online_TCP3.csv'
    action_name = 'A'
    reward_name = 'R'
    feature_name_curr = 'S1:S2:S3'
    feature_name_next = 'Sp1:Sp2:Sp3'
    feature_type = 'ActiveFlow:ActiveFlow:ActiveFlow'
    feature_weight = '0.2:0.2:0.2:0.2'
    start = 0
    length =300
    print 'training stage...'
    RL = KEMDRL()
    model = RL.train( input, action_name, reward_name, feature_name_curr, feature_name_next, feature_type, feature_weight, start, length)
    #print 'batch test stage'
    model2 = KEMDModel()
    model2.load('KEMD_model.txt')
    # #model2.predict(input, 400, 100)
    # #print 'single test stage ...'
    singleTestSuite = ["8:6:4"]
    # #print "Before optimization"
    # # tic = time.time()
    # # for i in range(0,1000):
    
    a=model2.predictSingle(singleTestSuite[0],0)
    print "The action is", a 
    
    # # toc = time.time()
    # # print "The time for a single flow prediction is", float(toc-tic)/1000
    # #print "After optimization"
    # # tic = time.time()
    # # for i in range(0,1000):
    # #a = model2.predictSingle_2(singleTestSuite[0],0)
    # #print a
    # # toc = time.time()
    # # print "The time for predicting one flow is", float(toc-tic)/1000
    # # print "The predicted flow size is", a  
   
    # # tic = time.time()
    # # for i in range(0,1000):
    print "The original model is:"
    print model2.getM()
    print "Online update"
    singleUpdateSuite = ["3:4:5:19:7:4:5:5"]
    for i in range(0,100):
        model2.update_online(singleUpdateSuite[0], 10000)
    print "The updated model is"
    print model2.getM()
    
   
                   
    