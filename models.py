import numpy as np
#-------------------------------------------------------------------------------------------
class LogisticRegression:
    def __init__(self,alpha=0.01,reg=0,eps=1e-5,max_iter=200):
        self.alpha=alpha
        self.reg=reg
        self.eps=eps
        self.max_iter=max_iter
    def sigmoid(self,z):
        z = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z))
    def fit(self,X,y):
        m,n=X.shape
        self.theta=np.zeros(n)
        it=0
        prev_theta=self.theta+5*self.eps
        
        while it<self.max_iter and np.linalg.norm(prev_theta-self.theta,ord=1)>self.eps:
            it+=1
            prev_theta=self.theta.copy()
            penalty=self.reg*self.theta
            penalty[0]=0.0
            self.theta=self.theta+self.alpha/m*X.T@(y-self.sigmoid(X@self.theta))-penalty
        if it<self.max_iter:
            print(f"Converged in {it} iterations")
        else:
            print(f"Failed to converge in {it} iterations")
    def predict(self,x):
        return self.sigmoid(x@self.theta)>=0.5
#-------------------------------------------------------------------------------------------
class KnnClassifier:
    def __init__(self,k,p=2):
        self.k=k
        self.p=p
    def fit(self,X,y):
        self.X=np.asarray(X)
        self.y=np.asarray(y)
    def predict(self,X):
        X = np.asarray(X)
        preds=[]
        for sample in X:
            dist=np.linalg.norm(self.X-sample,ord=self.p,axis=1)
            idx=np.argsort(dist)[:self.k]
            y_k=self.y[idx]
            unique_c,counts=np.unique(y_k,return_counts=True)
            max_count=np.max(counts)
            tied=unique_c[counts==max_count]
            if len(tied)>1:
                tied_dist=[]
                for c in tied:
                    tied_dist.append(np.sum(dist[idx[y_k==c]]))
                preds.append(tied[np.argmin(tied_dist)])
            else:
                preds.append(tied[0])
        return preds
#-------------------------------------------------------------------------------------------
class LinearRegression:
    def fit(self,X,y):
        self.X=np.asarray(X)
        self.y=np.asarray(y)
        self.theta=np.linalg.inv(X.T@X)@X.T@y
    def predict(self,X):
        X=np.asarray(X.copy())
        return X@np.reshape(self.theta,(self.theta.shape[0],1))
#-------------------------------------------------------------------------------------------
class Locally_weighted_LR:
    def __init__(self,tau=0.1):
        self.tau=tau
    def fit(self,X,y):
        self.X=np.asarray(X)
        self.y=np.asarray(y)
    def predict(self,X):
        m,n=self.X.shape
        X=np.asarray(X)
        l=X.shape[0]
        diff=np.reshape(X,(-1,l,n))-np.reshape(self.X,(m,-1,n))
        w=np.exp(-np.linalg.norm(diff,ord=2,axis=2)**2/(2*self.tau**2))#mxl
        W=np.apply_along_axis(np.diag,axis=0,arr=w).T #lxmxm
        XT_W=self.X.T@W
        theta = np.linalg.solve(XT_W @ self.X, np.reshape(XT_W @ self.y,(l,n,1))).squeeze()#lxn
        return np.sum(X*theta,axis=1)#lx1
#-------------------------------------------------------------------------------------------
class GLM:#Based on exponential  distribution
    def __init__(self,eps=1e-5,max_iter=100):
        self.eps=eps
        self.max_iter=max_iter
    def fit(self,X,y):
        self.X=np.asarray(X)
        self.y=np.asarray(y)
        m,n=self.X.shape
        self.theta=np.zeros(n)-0.1
        prev_theta=self.theta+2*self.eps
        it=0
        while np.linalg.norm(self.theta-prev_theta,ord=1)>=self.eps and it<self.max_iter:
            it+=1
            prev_theta=self.theta.copy()
            X_theta=self.X@self.theta
            
            gradient=self.X.T@(self.y+1/(X_theta))
            W=np.diag(1/(X_theta)**2)
            hessian=-X.T@W@X
            self.theta=self.theta-np.linalg.inv(hessian)@gradient
        print(f"Converged in {it} iterations")
    def predict(self,X):
        X=np.asarray(X)
        return -1/(X@self.theta)
#-------------------------------------------------------------------------------------------
class SoftMax:
    def __init__(self,eps=1e-5,max_iter=100,lr=0.1,l2=0):
        self.eps=eps
        self.max_iter=max_iter
        self.lr=lr
        self.l2=l2
    def fit(self,X,y):
        self.X=np.asarray(X)
        self.y=np.asarray(y)
        m,n=self.X.shape
        self.classes=np.unique(y)
        self.n_classes=len(self.classes)
        self.theta={c:np.zeros(n) for c in self.classes}
        
        it=0
        max_diff=2*self.eps
        while max_diff>=self.eps and it<self.max_iter:
            it+=1
            prev_theta=self.theta.copy()
            max_diff=0
            prev_theta=self.theta.copy()
            for c in self.classes:
                theta_c=self.theta[c]
                P_c=np.exp(X@theta_c)/np.sum([np.exp(X@prev_theta[j])for j in self.classes],axis=0)       
                self.theta[c]=theta_c-self.lr*((P_c-(y==c).astype('int32'))@self.X/m+self.l2*self.theta[c])
                diff=np.linalg.norm(self.theta[c]-prev_theta[c],ord=1)
                if diff>max_diff:
                    max_diff=diff
            
            #print(max_diff)
        print(f"total iterations: {it}")
    def predict(self,X):
        m,n=X.shape
        probs=np.zeros((m,self.n_classes))
        for i,c in enumerate(self.classes):
            theta_c=self.theta[c]    
            P_c=np.exp(X@theta_c)/np.sum([np.exp(X@self.theta[j])for j in self.classes],axis=0)
            probs[:,i]=P_c
        results=np.argmax(probs,axis=1)
        return self.classes[results]
#-------------------------------------------------------------------------------------------