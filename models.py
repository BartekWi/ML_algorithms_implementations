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
class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
    def IsLeaf(self):
        return self.value != None
class DecisionTree:
    def __init__(self,max_depth=50,min_samples=10,n_features=None,random_state=42):
        self.max_depth=max_depth
        self.min_samples=min_samples
        self.n_features=n_features
        self.root=None
        self.rng=np.random.RandomState(random_state)
    def fit(self,X,y):
        self.n_features=X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.X=np.asarray(X)
        self.y=np.asarray(y)
        self.root=self._grow_tree(self.X,self.y)
    def _grow_tree(self,X,y,depth=0):
        m,n=X.shape
        classes,counts=np.unique(y,return_counts=True)
        n_classes=len(classes)
        if depth>=self.max_depth or self.min_samples>len(y) or n_classes==1:
            return Node(value=classes[np.argmax(counts)])
        feature_idxs=self.rng.choice(n,self.n_features,replace=False)
        best_feature,best_thr=self._optimal_split(X,y,feature_idxs)
        if best_feature is None:
            return Node(value=classes[np.argmax(counts)])
        l_idxs,r_idxs=self._split(X[:,best_feature],best_thr)
        left=self._grow_tree(X[l_idxs],y[l_idxs],depth+1)
        right=self._grow_tree(X[r_idxs],y[r_idxs],depth+1)
        return Node(best_feature,best_thr,left,right)
    def _optimal_split(self,X,y,feature_idxs):
        best_ig=-1
        feature_idx,threshold=None,None
        for idx in feature_idxs:
            X_col=X[:,idx]
            thresholds=np.unique(X_col)
            for thr in thresholds:
                ig=self._calculate_IG(X_col,y,thr)
                if ig>best_ig:
                    best_ig=ig
                    feature_idx=idx
                    threshold=thr
        return feature_idx,threshold
    def _entropy(self,y):
            c_sum=0
            for c in np.unique(y):
                p_c=len(y[y==c])/len(y)
                c_sum+=p_c*np.log2(p_c)
            return -c_sum
    def _calculate_IG(self,X,y,threshold):      
        parent_e=self._entropy(y)
        l_idxs,r_idxs=self._split(X,threshold)
        if len(l_idxs) == 0 or len(r_idxs) == 0:
            return -1
        l_e=self._entropy(y[l_idxs])
        r_e=self._entropy(y[r_idxs])
        n_l=len(l_idxs)
        n_r=len(r_idxs)
        return parent_e-(n_l*l_e+n_r*r_e)/(n_l+n_r)
    def _split(self,X,threshold):
        return np.argwhere(X<=threshold).flatten(), np.argwhere(X>threshold).flatten()
    def _check_tree(self,x,node):
        if node.IsLeaf():
            return node.value
        if x[node.feature]<=node.threshold:
            return self._check_tree(x,node.left)
        return self._check_tree(x,node.right)
    def predict(self,X):
        return np.array([self._check_tree(x,self.root) for x in X])       
#-------------------------------------------------------------------------------------------
class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value
    def IsLeaf(self):
        return self.value != None
class RegressionTree:
    def __init__(self,max_depth=50,min_samples=10,n_features=None,random_state=42):
        self.max_depth=max_depth
        self.min_samples=min_samples
        self.n_features=n_features
        self.root=None
        self.rng=np.random.RandomState(random_state)
    def fit(self,X,y):
        self.n_features=X.shape[1] if not self.n_features else max(1,min(X.shape[1],self.n_features))
        self.X=np.asarray(X)
        self.y=np.asarray(y)
        self.root=self._grow_tree(self.X,self.y)
    def _grow_tree(self,X,y,depth=0):
        m,n=X.shape
        if depth>=self.max_depth or self.min_samples>len(y) or len(np.unique(y)) == 1:
            return Node(value=np.mean(y))
        feature_idxs=self.rng.choice(n,self.n_features,replace=False)
        best_feature,best_thr=self._optimal_split(X,y,feature_idxs)
        l_idxs,r_idxs=self._split(X[:,best_feature],best_thr)
        left=self._grow_tree(X[l_idxs],y[l_idxs],depth+1)
        right=self._grow_tree(X[r_idxs],y[r_idxs],depth+1)
        return Node(best_feature,best_thr,left,right)
    def _optimal_split(self,X,y,feature_idxs):
        best_ss=-1
        feature_idx,threshold=None,None
        for idx in feature_idxs:
            X_col=X[:,idx]
            thresholds=np.unique(X_col)
            for thr in thresholds:
                phi=self._calculate_phi(X_col,y,thr)
                if phi>best_ss:
                    best_ss=phi
                    feature_idx=idx
                    threshold=thr
        return feature_idx,threshold
    def _ss(self,y):
        if len(y)==0:
            return 0
        return np.sum((y-np.mean(y))**2)
    def _calculate_phi(self,X,y,threshold): 
        
        parent_ss=self._ss(y)
        l_idxs,r_idxs=self._split(X,threshold)
        if len(l_idxs) == 0 or len(r_idxs) == 0:
            return -1
        l_ss=self._ss(y[l_idxs])
        r_ss=self._ss(y[r_idxs])
        return parent_ss-l_ss-r_ss
    def _split(self,X,threshold):
        return np.argwhere(X<=threshold).flatten(), np.argwhere(X>threshold).flatten()
    def _check_tree(self,x,node):
        if node.IsLeaf():
            return node.value
        if x[node.feature]<=node.threshold:
            return self._check_tree(x,node.left)
        return self._check_tree(x,node.right)
    def predict(self,X):
        return np.array([self._check_tree(x,self.root) for x in X])          
#-------------------------------------------------------------------------------------------
class NaiveBayes:
    def fit(self,X,y):
        self.X=np.asarray(X)
        self.y=np.asarray(y)
        m,n=self.X.shape
        y1_count=np.sum(X[y==1])
        y0_count=np.sum(X[y==0])
        self.p_y1=len(y[y==1])/m
        self.p_y0=len(y[y==0])/m
        self.p_xy1=(np.sum(X[y==1],axis=0)+1)/(y1_count+n)
        self.p_xy0=(np.sum(X[y==0],axis=0)+1)/(y0_count+n)
    def predict(self,X):
        X=np.asarray(X)
        y1=np.sum(X*np.log(self.p_xy1),axis=1)+np.log(self.p_y1)
        y0=np.sum(X*np.log(self.p_xy0),axis=1)+np.log(self.p_y0)
        return np.argmax(np.column_stack((y0, y1)),axis=1)
#-------------------------------------------------------------------------------------------
class GDA:
    def fit(self,X,y):
        self.X=np.asarray(X)
        self.y=np.asarray(y)
        m,n=self.X.shape
        self.classes=np.unique(y)
        self.n_classes=len(self.classes)
        self.mus=np.zeros((self.n_classes,n))
        self.covs = np.zeros((self.n_classes, n, n))
        self.phis=np.zeros(self.n_classes)
        for i,c in enumerate(self.classes):
            self.phis[i]=len(y[y==c])/m
            self.mus[i,:]=np.mean(self.X[y==c],axis=0)
            self.covs[i, :, :] = np.cov(self.X[y==c].T, bias=True)
    def predict(self,X):
        X=np.asarray(X)
        probs=np.zeros((X.shape[0],self.n_classes))
        for i,c in enumerate(self.classes):
            cov = self.covs[i]
            term_1 = -0.5 * X.shape[1] * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov))
            diff = X - self.mus[i]
            term_2 = -0.5 * np.sum(diff@np.linalg.inv(cov) * diff, axis=1)
            probs[:,i]=term_1+term_2+np.log(self.phis[i])
        return self.classes[np.argmax(probs,axis=1)]