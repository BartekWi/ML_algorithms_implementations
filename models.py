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