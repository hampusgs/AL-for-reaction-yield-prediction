import os
import sys
import argparse
import numpy as np
import pandas as pd
import smurff
import time
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  roc_auc_score

class ActiveLearner():
    def __init__(self, dataset, train_file, test_file, acquisition, num_latent=4, threshold=20, model='mf'):
        self.save_prefix=train_file+'_'+test_file+'_'+acquisition+'/'
        self.num_latent=num_latent
        self.threshold=threshold
        self.suffix = 0
        self.data=pd.read_csv(dataset, header=0)
        self.train_indices=[]
        df=self.data
        names=list(df)
        self.train=pd.read_csv(train_file, header=0, names=names)
        for _,rows in self.train.iterrows():
            df_copy=df.copy()
            for name in names[:-1]:
                df_copy=df_copy.loc[(df[name]==int(rows[name]))]
            self.train_indices.extend(df_copy.index.tolist())
        self.train_indices=np.array(self.train_indices)
        self.size_matrix=len(self.data)
        self.index_total=np.arange(0,self.size_matrix)
        self.index_total=self.find_remaining_points(self.index_total,self.train_indices)
        self.test_indices=[]
        self.test=pd.read_csv(test_file, header=0, names=names)
        for _,rows in self.test.iterrows():
            df_copy=df.copy()
            for name in names[:-1]:
                df_copy=df_copy.loc[(df[name]==int(rows[name]))]
            self.test_indices.extend(df_copy.index.tolist())
        self.test_indices=np.array(self.test_indices)
        if model=='mf':        
            self.test_data = smurff.make_sparse(self.data.loc[
                self.data.index.isin(self.test_indices)],len(self.test_indices))
        if model=='rf':
            self.test_data = np.array(self.data.iloc[self.test_indices].values)
     
        self.index_total=self.find_remaining_points(self.index_total,self.test_indices)
        self.index_total_start=np.copy(self.index_total)
        print('Ratio of successful reactions at threshold:', np.count_nonzero(
            self.data[names[-1]].values > self.threshold, axis=0)/self.data.shape[0])
        self.eps=1e-8
        self.end=self.data.shape[1]
  
        os.makedirs(self.save_prefix,exist_ok=True)
    
    def find_remaining_points(self,A,B):
        #Returns all the elements in A not present in B
        sidx=B.argsort()
        idx = np.searchsorted(B,A,sorter=sidx)
        idx[idx==len(B)]= 0
        return A[B[sidx[idx]] != A]
    
    def mask_prediction(self, pred, masking_indices, model_type):
        if model_type == 'mf':
            mask=np.zeros(pred.shape)
            remaining = []
            red_data=self.data.columns[0:(len(self.data.columns)-1)]
            for i in masking_indices:
                new=self.data[red_data].values[i]
                remaining.append(new)
            remaining=np.array(remaining)
            for l in range(0,len(remaining)):
                mask_loc=[]
                for k in range(remaining.shape[1]):
                    mask_loc.append(remaining[l,k])
                mask[tuple(mask_loc)]=1
            return mask*pred
        if model_type == 'rf':
            mask=np.zeros(pred.shape)
            for i in masking_indices:
                    mask[i]=1
            return mask*pred

    def get_new_indices(self,pred,n, model_type):
        if model_type == 'mf':
            array_to_sort=(self.mask_prediction(pred, self.index_total, model_type))
            ordered_indices = self.largest_indices(np.absolute(array_to_sort),n)
            pos=np.unravel_index(ordered_indices,pred.shape)
            index_to_add=[]
            for i in range(len(pos[0])):
                a=self.data
                for j in range(0,len(a.columns)-1):           
                    a=a[a[a.columns[j]]==pos[j][i]]
                index_to_add.append(a.index[0])
            index_to_add=np.array(index_to_add)
            return index_to_add
        if model_type == 'rf':
            array_to_sort=(self.mask_prediction(pred, model.index_total, model_type))
            ordered_indices = self.largest_indices(np.absolute(array_to_sort),1)
            index_to_add=np.array(ordered_indices[0])
            return index_to_add


    
    def binary_uncertainty(self,input):
        distances=1/(self.eps+np.absolute(0.5-input))
        return distances
    
    def random_selection(self,input, model_type):
        if model_type == 'rf':
            return np.random.rand(len(input))
        else:
            return np.random.random_sample(tuple(model.test_data.shape))
    
    def increment_save_dir(self):
        self.suffix += 1
        self.save_name = self.save_prefix + '/active_learning_iter_{}'.format(self.suffix)
        
    def largest_indices(self,ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        a=-flat
        b=np.random.random(a.size)
        indices = np.lexsort((b,a))
        c=indices[:n]
        return c       
    
    def Macau(self,train_indices):
        """Run the training Macau
            """
        data=model.data.loc[model.data.index.isin(train_indices)]
        train_data=smurff.SparseTensor(data,shape=model.test_data.shape)
        self.increment_save_dir() 
        session = smurff.MacauSession(Ytrain=train_data,
                                Ytest=model.test_data,
                                num_latent=self.num_latent,
                                burnin=200,
                                nsamples=1000,
                                save_freq=5,
                                verbose=True,
                                save_name=model.save_name)
        
        session.addTrainAndTest(train_data, model.test_data, smurff.ProbitNoise(self.threshold))
        return session

    def calc(self,train_indices):
            """Run the training Macau
                """
            data=model.data.loc[model.data.index.isin(train_indices)]
            
            data=model.data.loc[
                model.data.index.isin(train_indices)]
            train_data=np.array(data.values)
            rf_run=RandomForestClassifier()
            train_X=train_data[:,0:self.end-1]
            train_Y=train_data[:,self.end-1]
            train_Y=(train_Y>=20)*1
            #fake first iteration if all values has same label
            if np.all(train_Y==1):
                train_Y[0]=0
            if np.all(train_Y==0):
                train_Y[0]=1
            rf_run.fit(train_X,train_Y)
        
            return rf_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'train a matrix factorization model using SMURFF')
    parser.add_argument("--threshold", "-th", help='threshold for true classification',\
         default=20, type=float, required=False)
    parser.add_argument("--data", "-d",\
         help='Which dataset to use', required=False)
    parser.add_argument("--acquisition", "-a",\
        help='Which acquisition function to use',\
            default='both', choices=['uncertainty','both','random'], required=False)
    parser.add_argument("--model", "-m",\
        help='Which acquisition function to use',\
            default='rf', choices=['mf','rf'], required=False)
    parser.add_argument("--num_latent",
                         help='latent features', default=4,
                          type=int, required=False)
    parser.add_argument("--starting_file", "-sf",
                         help='Start training points', type=str, required=False)
    parser.add_argument("--test_file", "-tf",
                         help='Starting test points', type=str, required=False)
    parser.add_argument("--end_size", "-es",
                         help='Size of final input per run', default=20,
                          type=int, required=False)
    parser.add_argument("--n_splits", "-ns",
                         help='Number of splits', default=5,
                          type=int, required=False)
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    start = time.time()
    model= ActiveLearner(args['data'],
                            args['starting_file'],
                            args['test_file'],
                            args['acquisition'],
                            args['num_latent'],
                            args['threshold'],
                            args['model'])
    print('Initializing model with training set:', args['data'])

    start_size=len(model.train_indices)
    n_iter=(args['end_size'] - start_size)
    print('Run started, {} splits starting at {} points, going to {} points'.format(args['n_splits'],start_size ,args['end_size']))
    chosen_acqs=[args['acquisition']]
    if args['acquisition']=='both':
        chosen_acqs=['random','uncertainty']
    
    for a in chosen_acqs:
        for w in range(0,args['n_splits']):
            if args['model']=='rf':
                test_X=model.test_data[:,0:model.end-1]
                test_Y=model.test_data[:,model.end-1]
                test_Y=(test_Y>=20)*1
        
            
            train_indices=np.copy(model.train_indices)
            model.index_total=np.copy(model.index_total_start)
            auroc_run=[]
            for u in range(0,n_iter):
                sys.stdout.flush()
                if args['model']=='mf':
                    session = model.Macau(train_indices)
                    test=session.run()
                    auroc=smurff.calc_auc(test,20)
                    predict_session = session.makePredictSession()
                    result=predict_session.predict_all()
                    numbers=np.zeros(np.sum(result,axis=0).shape)
                    all_indices=np.arange(0,model.size_matrix)
                    pred= (expit(result))
                    n_positive=np.sum(pred>=0.5,axis=0)/200
                    np.save(model.save_prefix+'prob_tensor_split_{}_iter_{}_{}.npy'.format(w,u,a),n_positive)
                    if a == 'uncertainty':
                        acq_score=model.binary_uncertainty(n_positive)
                    if a == 'random':
                        acq_score=model.random_selection(n_positive, args['model'])
  
                
                if args['model']=='rf':
                    session = model.calc(train_indices)
                    
                    all_data=model.data.values
                    all_X=all_data[:,0:model.end-1]
                    all_Y=all_data[:,model.end-1]
                
                    pred_rf_100=session.predict_proba(all_X)
                    test_X=model.test_data[:,0:model.end-1]
                    test_Y=model.test_data[:,model.end-1]

                    test_Y=(test_Y>=20)*1
                    pred_rf=session.predict_proba(test_X)
                    auroc=roc_auc_score(test_Y, pred_rf[:,1])
                    features=session.feature_importances_
                    print('Feature Importances:', features)
                    print('Auroc:', auroc)
                    

                    
                    np.save(model.save_prefix+'prob_tensor_split_{}_iter_{}_{}.npy'.format(w,u,a),pred_rf_100)
                    if a == 'uncertainty':
                        acq_score=model.binary_uncertainty(pred_rf_100[:,1])
                    if a == 'random':
                        acq_score=model.random_selection(pred_rf_100[:,1], args['model'])
                    
                
                new_index=model.get_new_indices(acq_score, 1, args['model'])
                
                train_indices=np.append(train_indices,new_index)
                model.index_total=model.find_remaining_points(model.index_total,train_indices)
                print('Split {},  Iter {} finished. Acqusition: {}, Auroc: {}'.format(w,u,a,auroc))
                auroc_run.append(auroc)
            np.save(model.save_prefix+'auroc_split_{}_{}.npy'.format(w,a),np.array(auroc_run))
            
            
                
