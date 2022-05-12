def plot_learning_curve(estimator,title,x,y,ax=None,ylim=None,
                        cv=None,
                        n_jobs=None):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_sizes,train_scores,test_scores=learning_curve(estimator,x,y,shuffle=True,cv=cv,
                                                        # random_state=420,
                                                        n_jobs=n_jobs
                                                        )
    if ax==None:
        ax=plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("training examples")
    ax.set_ylabel("score")
    ax.grid()
    ax.plot(train_sizes,np.mean(train_scores,axis=1),'o-',color='r',label='Train score')
    ax.plot(train_sizes,np.mean(test_scores,axis=1),'o-',color='g',label='Test score')
    ax.legend(loc='best')
    return ax