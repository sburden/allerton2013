"""
Nash equilibria in continuous games

jac - func - numerically approximate Jacobian of a map
descent - func - gradient descent

by Sam Burden, Lily Ratliff, Berkeley 2013
"""

import time

import numpy as np
import scipy as sp
import scipy.linalg
import pylab as plt
import matplotlib as mpl

np.set_printoptions(precision=2)

dbg = False
import sys
import os

def aunique(a):
  a = a[np.logical_not(np.all(np.isnan(a),axis=1))]
  ind = np.lexsort(a.T)
  return a[ind[np.concatenate(([True],np.any(a[ind[1:]]!=a[ind[:-1]],axis=1)))]]

def central(f,x,fx,d):
  """
  df = central()  compute central difference 

  df = 0.5*(f(x+d) - f(x-d))/np.linalg.norm(d)
  """
  return 0.5*(f(x+d) - f(x-d))/np.linalg.norm(d)

def forward(f,x,fx,d):
  """
  df = forward()  compute forward difference 

  df = (f(x+d) - fx)/np.linalg.norm(d)
  """
  return (f(x+d) - fx)/np.linalg.norm(d)

def jac(f, x, fx=None, d=1e-6, D=None, diff=forward):
  """
  .jac  Numerically approximate Jacobian to f at x

  Inputs:
    f : R^n --> R^m
    x - n vector
    d - scalar or (1 x n) - displacement in each coordinate
  (optional)
    fx - m vector - f(x)
    D - k x n - directions to differentiate (assumes D.T D invertible)
    diff - func - numerical differencing method

  Outputs:
    Df - m x n - Jacobian of f at x
  """
  if fx is None:
    fx = f(x)
  if D is None:
    Df = map(lambda dd : diff(f,x,fx,dd), list(d*np.identity(len(x))))
  else:
    Df = map(lambda dd : diff(f,x,fx,dd), list(D))

  return np.array(Df).T

def controllability(A,B):
  """
  controllability matrix of the pair (A,B)

  Inputs:
    A - n x n 
    B - n x m

  Outputs:
    C - n x (n*m) 
  """
  assert A.shape[0] == A.shape[1] # A is n x n
  assert A.shape[0] == B.shape[0] # B is n x m
  C = [B]
  for n in range(A.shape[0]):
    C.append( np.dot(A, C[-1]) )
  return np.hstack(C)

def controllable(A,B,eps=1e-3):
  """
  controllable  test controllability of the pair (A,B)
                for the LTI system  x' = A x + B u

  Inputs:
    A - n x n 
    B - n x m

  Outputs:
    bool - controllable with threshold eps
  """
  C = controllability(A,B)
  _,s,_ = np.linalg.svd(C)
  return np.all( s > eps )

def observability(A,C):
  """
  observability matrix of the pair (A,C)

  Inputs:
    A - n x n 
    C - m x n

  Outputs:
    O - (n*m) x n
  """
  assert A.shape[0] == A.shape[1] # A is n x n
  assert A.shape[0] == C.shape[1] # C is m x n
  return controllability(A.T,C.T).T

def observable(A,C,eps=1e-3):
  """
  observable  test observability of the pair (A,C)
              for the LTI system  x' = A x, y = C x

  Inputs:
    A - n x n 
    C - m x n

  Outputs:
    bool - observable with threshold eps
  """
  O = observability(A,C)
  _,s,_ = np.linalg.svd(O)
  return np.all( s > eps )

def PSD(n,sqrt=False):
  """
  PSD  compute random positive semidefinite matrix

  Inputs:
    n - int - dimension of matrix
    (optional)
    sqrt - bool - whether to return S such that Q = np.dot( S.T, S)

  Outputs:
    Q - n x n - Q = Q^T,  spec Q \subset R^+
  """
  H = np.random.randn(n,n)
  d,u = np.linalg.eig(H + H.T)
  S = np.dot( u, np.dot( np.diag( np.sqrt( d*np.sign(d) ) ), u.T ) )
  if sqrt:
    return np.dot(S.T, S), S
  else:
    return np.dot(S.T, S)

def euler(h, K, f, x0, u):
  """
  euler  Compute forward Euler approximation to ODE

  (*)  x'(t) = f(t, x(t), u(t)),  x(0) = x0

  Inputs:
    h - scalar - step size
    K - int - number of steps
    f : R x X x R^m --> X,  X a vector space
    x0 - initial state vector
    u - K x m - control at each timestep

  Outputs:
    x - K x n - trajectory satisfying (*)
  """
  x0 = np.array(x0)
  x  = [x0]
  for k in range(K-1):
    x.append(x[k] + h*f(k*h, x[k], u[k]))
  return x

def descent(x0, J, d=None, 
            eps=5e-2, Jtol=-np.inf, a=lambda a,b,c,**kwds : 1e-0, 
            xnorm=lambda x : np.sqrt(np.sum(x * x)), 
            dnorm=lambda d : np.sqrt(np.sum(d * d)), 
            jmax=200, xmax=1e6, trj=None, debug=False):
  """
  .descent  Gradient descent to multi-objective stationary points

  Inputs
    x0 in R^n - initial guess for stationary point
    J : R^n --> R^m - multiple objective functions
    d : R^n --> R^(m x n) - descent directions
  (optional)
    eps - scalar - threshold on norm of gradient for stationary point
    Jtol - scalar - threshold on value of objectives for stationary point
    a  : (J,x,d) |--> scalar  - stepsize
    xnorm : R^n --> R - norm for decision variable
    dnorm : R^n --> R - norm for descent direction
    jmax - integer - maximum number of iterations
    xmax - scalar  - largest allowable state
    trj - list - descent sequence
    debug - bool - debugging flag

  Outputs
    x - n vector - stationary point, i.e.  || D(f,x) || < eps
  """
  # initial point
  x = x0
  # objectives
  Jx = J(x)
  # descent direction
  dx = d(x)
  if debug:
    print '(descent)  x = %s, J(x) = %s' % (x,Jx)
    print '         d(x) = %s' % (dx.sum(axis=0))
    sys.stdout.flush()
  # loop over descent iterations
  for j in range(jmax):
    # fail if function or gradient are undefined
    if np.any(np.isnan(Jx)):
      if debug:
        print '(descent)  J(x) has nan(s)'
        assert dbg
      return np.nan*x,j
    if np.any(np.isnan(dx)):
      if debug:
        print '(descent)  d(x) has nan(s)'
        assert dbg
      return np.nan*x,j
    # succeed at stationary point
    if dnorm(dx) < eps or np.all(Jx < Jtol):
      return x,j
    # compute stepsize
    aa = a(x,J,dx,Jx=Jx,debug=debug)
    # fail if stepsize cannot be chosen
    if np.isnan(aa):
      if debug:
        print '(descent)  stepsize is nan'
        assert dbg
      return np.nan*x,j
    # descend
    x = x + aa * dx.sum(axis=0)
    if trj is not None:
      trj.append(x)
    # fail if point is unbounded
    if xnorm(x) >= xmax:
      if debug:
        print '(descent)  xnorm(x) >= xmax'
        assert dbg
      return np.nan*x,j
    # evaluate function
    Jx = J(x)
    # descent direction
    dx = d(x)
    if debug:
      print 'j = %3d, a = %0.2e, x = %s, J(x) = %s' % (j,aa,x,J(x))
      print '         d = %s' % (dx.sum(axis=0))
      sys.stdout.flush()
  # descent failed to converge
  if debug:
    print '(descent)  j >= jmax'
    #assert dbg
  return x,j

class DG(object):
  """
  Differential Game
  """

  def __init__(self,L,F,J,DxF=None,DuF=None,DxJ=None):
    """
    dg = DG(...)

    Inputs:
      L - Z^m - dimensions of player inputs; l = L.sum()
      F : R x R^n x R^l --> R^n - vector field
      time t in R, state x in R^n, control u_i in R^l_i, l_1+...+l_m = L.sum()
      J : R^n --> R^m - player utilities
      (optional)
      DxF : R x R^n x R^l --> R^(n x n) - derivative of F w.r.t. x
      DuF : R x R^n x R^l --> R^(n x l) - derivative of F w.r.t. u
      DxJ : R^n --> R^(m x n) - derivative of J w.r.t. x

    Notes:
      n = dim x, l.sum() = dim u, m = len(L) = # of players
    """
    self.L = np.asarray(L)
    self.F = F
    self.J = J
    if DxF is None:
      DxF = lambda t,x,u : jac( lambda x_ : F(t,x_,u), x )
    if DuF is None:
      DuF = lambda t,x,u : jac( lambda u_ : F(t,x,u_), u )
    if DxJ is None:
      DxJ = lambda x : jac( J, x )
    self.DxF = DxF
    self.DuF = DuF
    self.DxJ = DxJ
    # encode player control input
    self.mask = sp.linalg.block_diag(*[np.ones(l) for l in L])

  def state(self,h,K,x0,u):
    """
    x = state(...)

    Inputs:
      h - scalar - timestep
      K - int - # of steps
      x0 - R^n - initial state
      u - R^(K x l) - control inputs

    Outputs:
      x - R^(K x n) - state
    """
    # state
    x = np.array( euler(h, K, self.F, x0, u) )
    return x

  def costate(self,h,K,x,u):
    """
    P = costate(...)

    Inputs:
      h - scalar - timestep
      K - int - # of steps
      x - R^(K x n) - state
      u - R^(K x l) - control inputs

    Outputs:
      P - R^(K x m x n) - costate for each player
    """
    # problem dimensions
    n = x.shape[1]; l = self.L.sum()
    # combination of state and control
    xu = np.hstack((x,u))
    # costate dynamics
    dP = lambda t,P,xu : np.dot( P, self.DxF(t,xu[:n],xu[n:]) )
    # final condition
    PT = self.DxJ(x[-1])
    # costate
    P = np.asarray( euler(h, K, dP, PT, xu[::-1])[::-1] )
    return P

  def DuJ(self,h,K,x0,u):
    """
    d = DuJ(...)

    Inputs:
      h - scalar - timestep
      K - int - # of steps
      x0 - R^n - initial state
      u - R^(K x l) - control inputs

    Outputs:
      d - R^(K x m x l) - derivative
    """
    # state
    x = self.state(h,K,x0,u)
    # costate
    P = self.costate(h,K,x,u)
    # pull back costate into R^m via DuF
    d = np.asarray([np.dot(P[k],self.DuF(h*k,x[k],u[k])) for k in range(K)])/K
    # restrict search to player's control input
    return d * self.mask

  def nlp(self,h,K,x0,u0,jmax=40,eps=1e-4,dbg=False,method='steep'):
    """
    u = nlp(...)  find local minima using nonlinear programming (NLP)

    Inputs:
      h - scalar - timestep
      K - int - # of steps
      x0 - R^n - initial state
      u0 - R^(K x l) - control inputs

    Outputs:
      u - R^(K x l) - control inputs
    """
    # problem dimensions
    l = self.L.sum(); m = self.L.size
    # encode optimization problem data
    u0_ = u0.reshape(K*l,order='F')
    J_ = lambda u_ : self.J(euler(h,K,self.F,x0,u_.reshape(K,l,order='F'))[-1])
    def DJ_(u_):
      d_ = self.DuJ(h,K,x0,u_.reshape(K,l,order='F'))
      d = np.asarray([d_[:,i,:].reshape(K*l,order='F') for i in range(m)])
      return -d
    self.J_ = J_; self.DJ_ = DJ_
    a = lambda a,b,c,**kwds : 1e-0 # constant stepsize
    self.res = descent(u0_, J_, d=DJ_, a=a, jmax=jmax, eps=eps, debug=dbg)
    u_ = self.res[0]
    u = u_.reshape(K,l,order='F')
    return u
    
def do_circ(seed, maxiter, dbg, sfxs=['pdf']):

  print 'seed = %s' % seed
  np.random.seed(seed)

  st = time.time()
  if 1:
    n = 2
    N = 4000
    #N = 50
    N = int(np.sqrt(N))**2
    K = 100

    I = np.identity(n)

    fi = 'nl_s%d_N%d' % (seed,N)
    
    alpha = 1.0*np.ones(n)*[1.-.00,1.+0.05]
    #alpha = 1.0*np.ones(n)*[1.-.00,1.+0.25]
    #alpha = 1.0*np.ones(n)*[1.-.00,1.+0.45]
    #alpha = 1.0*np.ones(n)*[1.-.00,1.+0.75]
    #alpha = 1.0*np.ones(n)*[1.-.00,1.+1.05]
    psi = 0.

    def J(th):
      a = + np.sum( np.cos( ( th[:,np.newaxis] - th ) ), axis=1 ) 
      b = - np.cos( th - psi )
      return alpha * a + b

    def d(th):
      a = - np.sum( np.sin( ( th[:,np.newaxis] - th ) ), axis=1 ) 
      b = np.sin( th - psi )
      return np.diag( -(alpha * a + b) )

    def DiJ(th):
      a = - np.sum( np.sin( ( th[:,np.newaxis] - th ) ), axis=1 ) 
      b = np.sin( th - psi )
      return alpha * a + b

    def DiiJ(th):
      a = - np.sum( (1.-I)*np.cos((th[:,np.newaxis]-th)), axis=1 ) 
      b = np.cos( th - psi )
      return alpha * a + b
  
    def Dw(th):
      return ( np.diag(DiiJ(th)) 
               + (1.-I) * np.cos( ( th[:,np.newaxis] - th ) ) )

    a = lambda a,b,c,**kwds : 1e-1

    pmpi = lambda th : np.mod(th + np.pi, 2*np.pi) - np.pi
    mdpi = lambda th : np.mod(th, 2*np.pi)

    import matplotlib.cm as cm
    fs = 16

    fig = plt.figure(1,figsize=(6.5,6)); fig.clf()
    ax = fig.add_subplot(111); ax.grid('on')
    ax.set_xlim(-np.pi,np.pi); ax.set_ylim(0.,2*np.pi); ax.set_aspect('equal')

    ax.set_xlabel(r'$\theta_1$',fontsize=fs); ax.set_ylabel(r'$\theta_2$',fontsize=fs)
    ax.set_title(r'$\alpha_1 = %0.2f, \alpha_2 = %0.2f$' 
                 % (alpha[0],alpha[1]),fontsize=fs)

    tix1 = np.linspace(-np.pi,np.pi,5)
    lbl1 = [r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$']
    ax.set_xticks(tix1); ax.set_xticklabels(lbl1,fontsize=fs)
    tix2 = np.linspace(0.,2*np.pi,5)
    lbl2 = [r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$']
    ax.set_yticks(tix2); ax.set_yticklabels(lbl2,fontsize=fs)

    NN=1000; eps = 1e-6; dth = 1.0; lw = 4
    th1 = np.linspace(-2*np.pi,2*np.pi,NN)
    th2 = th1 + (-np.pi + np.arcsin(np.sin(th1)/alpha[0]))
    th1[(np.abs(np.diff(pmpi(th1))) > dth).nonzero()] = np.nan
    th2[pmpi(np.abs(th2)) < eps] = np.nan
    ax.plot(pmpi(th1),mdpi(th2),'k',lw=lw,zorder=2,label=r'$D_1J_1 = 0$')
    th2 = th1 - np.arcsin(np.sin(th1)/alpha[0])
    #th1[(np.abs(np.diff(pmpi(th1))) > dth).nonzero()] = np.nan
    th2[pmpi(np.abs(th2)) < eps] = np.nan
    th2[(np.abs(np.diff(mdpi(th2))) > dth).nonzero()] = np.nan
    ax.plot(pmpi(th1),mdpi(th2),'k',lw=lw,zorder=2)
    fig.canvas.draw()

    th2 = np.linspace(-2*np.pi,2*np.pi,NN)
    th1 = th2 + (-np.pi + np.arcsin(np.sin(th2)/alpha[1]))
    th2[(np.abs(np.diff(mdpi(th2))) > dth).nonzero()] = np.nan
    th1[pmpi(np.abs(th1)) < eps] = np.nan
    ax.plot(pmpi(th1),mdpi(th2),'k--',lw=lw,zorder=2,label=r'$D_2J_2 = 0$')
    th1 = th2 - np.arcsin(np.sin(th2)/alpha[1])
    th2[(np.abs(np.diff(mdpi(th2))) > dth).nonzero()] = np.nan
    th1[pmpi(np.abs(th1)) < eps] = np.nan
    ax.plot(pmpi(th1),mdpi(th2),'k--',lw=lw,zorder=2)
    fig.canvas.draw()

    two = 2**np.arange(n)[::-1]
    if 1:
      th01 = np.linspace(-np.pi,np.pi,np.sqrt(N))
      th02 = np.linspace(0,2*np.pi,np.sqrt(N))
      Th0 = np.asarray(np.meshgrid(th01,th02)).reshape(2,N).T
    else:
      Th0 = (np.random.rand( N, n ) - .5) * 4*np.pi
    Th = np.nan*Th0
    B = np.nan*np.zeros(N)
    Iter = []
    for k,th0 in enumerate(Th0):
      th,iter = descent( th0, J, d, a=a, eps=2e-5, jmax=maxiter)
      Th[k] = [pmpi(th[0]),mdpi(th[1])]
      Iter.append(iter)

      th_ = [pmpi(th[0]),mdpi(th[1])]
      th0_ = [pmpi(th0[0]),mdpi(th0[1])]
      ax.plot(th0_[0],th0_[1],'k.',ms=10,mew=0.,zorder=-2)

      if k % K == K - 1:
        print '%d / %d took %0.2f sec' % (k,N,time.time() - st)
        fig.canvas.draw()

    print 'iters = %0.1f +/- %0.1f' % (np.mean(Iter), np.std(Iter))
    jnan = np.any(np.isnan(Th),axis=1).nonzero()
    nash0 = aunique(np.round(Th,1))
    print 'nash0 = %s' % nash0
    ax.plot(-2*np.pi,0,'o',color='k',ms=20,mew=1.,mec='w',
            label=r'$\omega = 0, D\omega > 0$')
    nash = np.asarray([na for na in nash0 if np.all(np.linalg.eigvals(Dw(na)) > 0)])
    print 'nash = %s' % nash
    print [np.linalg.eigvals(Dw(na)) for na in nash]
    cols = mpl.cm.hsv(np.linspace(0.,1.,nash.shape[0]))
    if len(nash) == 2:
      cols = np.asarray([[0.,0.,1.],[0.,1.,0.]])
    if len(nash) == 1:
      cols = np.asarray([[1.,0.,0.]])
    Th0_ = pmpi(Th0); Th_ = pmpi(Th)
    Nash = np.dstack(nash)
    B_ = np.argmin( np.sum( np.abs(np.exp(1.j*Th_)[...,np.newaxis] - np.exp(1.j*Nash))**2, axis=1), axis=1)
    for j,(na,col) in enumerate(zip(nash,cols)):
      ax.plot(na[0],na[1],'o',color=col,ms=30,mew=2.,mec='k',zorder=1)
      ax.contourf(th01,th02,B_.reshape((np.sqrt(N),np.sqrt(N))),1,zorder=-1,levels=[j-.5,j+.5],colors=[[c if c <= 1. else 1. for c in (.6+col**1e-0)]])

    ax.legend(loc=4,ncol=1,numpoints=1)
    fig.canvas.draw()

    # save figure
    for sfx in sfxs:
      if not os.path.exists(sfx):
        os.mkdir(sfx)
      f = os.path.join(sfx,fi+'.'+sfx)
      print 'saving %s' % f
      fig.savefig(f,dpi=300)

    # store data
    dat = dict(alpha=alpha,psi=psi,n=n,N=N,Th0=Th0,Th=Th,B=B,Iter=Iter,
             jnan=jnan,nash=nash)
    if not os.path.exists('npz'):
      os.mkdir('npz')
    f = os.path.join('npz',fi)
    print 'saving %s' % f
    np.savez(f,**dat)

    fig = plt.figure(1,figsize=(6.5,6)); fig.clf()
    ax = fig.add_subplot(111); ax.grid('on')
    ax.set_xlim(-np.pi,np.pi); ax.set_ylim(0.,2*np.pi); ax.set_aspect('equal')

    ax.set_xlabel(r'$\theta_1$',fontsize=fs); ax.set_ylabel(r'$\theta_2$',fontsize=fs)
    ax.set_title(r'$\alpha_1 = %0.2f, \alpha_2 = %0.2f$' 
                 % (alpha[0],alpha[1]),fontsize=fs)

    tix1 = np.linspace(-np.pi,np.pi,5)
    lbl1 = [r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$']
    ax.set_xticks(tix1); ax.set_xticklabels(lbl1,fontsize=fs)
    tix2 = np.linspace(0.,2*np.pi,5)
    lbl2 = [r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$']
    ax.set_yticks(tix2); ax.set_yticklabels(lbl2,fontsize=fs)

    NN=1000; eps = 1e-6; dth = 1.0; lw = 4
    th1 = np.linspace(-2*np.pi,2*np.pi,NN)
    th2 = th1 + (-np.pi + np.arcsin(np.sin(th1)/alpha[0]))
    th1[(np.abs(np.diff(pmpi(th1))) > dth).nonzero()] = np.nan
    th2[pmpi(np.abs(th2)) < eps] = np.nan
    ax.plot(pmpi(th1),mdpi(th2),'k',lw=lw,zorder=2,label=r'$D_1J_1 = 0$')
    th2 = th1 - np.arcsin(np.sin(th1)/alpha[0])
    #th1[(np.abs(np.diff(pmpi(th1))) > dth).nonzero()] = np.nan
    th2[pmpi(np.abs(th2)) < eps] = np.nan
    th2[(np.abs(np.diff(mdpi(th2))) > dth).nonzero()] = np.nan
    ax.plot(pmpi(th1),mdpi(th2),'k',lw=lw,zorder=2)
    fig.canvas.draw()

    th2 = np.linspace(-2*np.pi,2*np.pi,NN)
    th1 = th2 + (-np.pi + np.arcsin(np.sin(th2)/alpha[1]))
    th2[(np.abs(np.diff(mdpi(th2))) > dth).nonzero()] = np.nan
    th1[pmpi(np.abs(th1)) < eps] = np.nan
    ax.plot(pmpi(th1),mdpi(th2),'k--',lw=lw,zorder=2,label=r'$D_2J_2 = 0$')
    th1 = th2 - np.arcsin(np.sin(th2)/alpha[1])
    th2[(np.abs(np.diff(mdpi(th2))) > dth).nonzero()] = np.nan
    th1[pmpi(np.abs(th1)) < eps] = np.nan
    ax.plot(pmpi(th1),mdpi(th2),'k--',lw=lw,zorder=2)

    ax.legend(loc=4,ncol=1,numpoints=1)
    fig.canvas.draw()

    # save figure
    for sfx in sfxs:
      if not os.path.exists(sfx):
        os.mkdir(sfx)
      f = os.path.join(sfx,fi+'_0.'+sfx)
      print 'saving %s' % f
      fig.savefig(f,dpi=300)

    for j,(na,col) in enumerate(zip(nash,cols)):
      ax.plot(na[0],na[1],'o',color=col,ms=30,mew=2.,mec='k',zorder=1)
    ax.plot(-2*np.pi,0,'o',color='k',ms=20,mew=1.,mec='w',
            label=r'$\omega = 0, D\omega > 0$')
    ax.legend(loc=4,ncol=1,numpoints=1)
    fig.canvas.draw()

    # save figure
    for sfx in sfxs:
      if not os.path.exists(sfx):
        os.mkdir(sfx)
      f = os.path.join(sfx,fi+'_1.'+sfx)
      print 'saving %s' % f
      fig.savefig(f,dpi=300)

    z = np.exp(1.j*np.linspace(0.,2*np.pi,9))*np.pi/4.
    Th0 = np.vstack(([na + np.vstack((z.real,z.imag)).T for na in nash]))
    for th0 in Th0:
      plt.plot(th0[0],th0[1],'x',ms=8,mew=4.,mec='k',zorder=-2)
    fig.canvas.draw()

    # save figure
    for sfx in sfxs:
      if not os.path.exists(sfx):
        os.mkdir(sfx)
      f = os.path.join(sfx,fi+'_2.'+sfx)
      print 'saving %s' % f
      fig.savefig(f,dpi=300)

    for th0 in Th0:
      trj = [th0]
      descent( th0, J, d, a=a, eps=2e-5, jmax=maxiter, trj=trj)
      trj = np.asarray(trj)
      b = np.argmin( np.sum( np.abs(np.exp(1.j*trj[-1]) - np.exp(1.j*nash))**2, axis=1))
      plt.plot(th0[0],th0[1],'x',ms=8,mew=4.,mec=cols[b],zorder=-2)
      plt.plot(trj[:,0],trj[:,1],lw=2.,color=cols[b],zorder=-2)
    fig.canvas.draw()

    # save figure
    for sfx in sfxs:
      if not os.path.exists(sfx):
        os.mkdir(sfx)
      f = os.path.join(sfx,fi+'_3.'+sfx)
      print 'saving %s' % f
      fig.savefig(f,dpi=300)

    for j,(na,col) in enumerate(zip(nash,cols)):
      ax.contourf(th01,th02,B_.reshape((np.sqrt(N),np.sqrt(N))),1,zorder=-1,levels=[j-.5,j+.5],colors=[[c if c <= 1. else 1. for c in (.6+col**1e-0)]])
    fig.canvas.draw()

    # save figure
    for sfx in sfxs:
      if not os.path.exists(sfx):
        os.mkdir(sfx)
      f = os.path.join(sfx,fi+'_4.'+sfx)
      print 'saving %s' % f
      fig.savefig(f,dpi=300)

  print '%0.2f sec' % (time.time() - st)

def do_nlp(seed,maxiter,dbg,sfxs=['png','pdf']):

  print 'seed = %s' % seed
  np.random.seed(seed)

  n = 2; L = [1,1]; l = sum(L); m = len(L)
  print 'n = %d, L = %s' % (n,L)

  A = np.random.randn(n,n)
  a = np.abs(np.linalg.eig(A)[0]).max()
  B_= [np.random.randn(n,l_i) for l_i in L]
  B = np.hstack([B_[i] for i in range(m)])

  Q_= [PSD(n) for i in range(m)]
  Q = np.vstack(Q_)
  R_= [[PSD(l_j) for l_i in L] for l_j in L]
  RI= [[np.linalg.inv(R_[j][i]) for i in range(m)] for j in range(m)]
  Rj= [sp.linalg.block_diag(*[R_[i][j] for i in range(m)]) for j in range(m)]
  R = np.vstack(Rj)
  T = 1.

  x0 = np.random.randn(n); x0 = x0 / np.linalg.norm(x0)

  print 'A'
  print A
  print np.linalg.eigvals(A)
  print 'B'
  print B
  print 'Q_'
  print Q_
  print 'R_'
  print R_
  print 'x0 = %s' % x0
  print 'T = %0.2f' % T

  assert np.any([np.all([controllable(A,B_[i]),
                         observable(A,sp.linalg.sqrtm(Q_[i]))]) 
                 for i in range(m)]),'(A,Bi,sqrt(Qi)) controllable, observable'

  # finite-time LQ game
  # step sizes
  H = np.logspace(np.log10(5e-1),np.log10(2e-3),10)
  # relative error figure
  fi = 'lq_s%d_N%d' % (seed,H.size)
  e = np.nan*H
  fs = 16; lw = 4.; ms=20
  fig0 = plt.figure(0,figsize=(7.0,6)); fig0.clf()
  ax0 = fig0.add_subplot(111); ax0.grid('on')
  ax0.set_xlabel(r'Number of samples in time discretization',fontsize=fs)
  ax0.set_ylabel(r'Relative error, $\|u_N - u_N^{LQ}\| / \|u_N^{LQ}\|$',fontsize=fs)
  ax0.tick_params(axis='both', which='major', labelsize=fs)
  ax0.tick_params(axis='both', which='minor', labelsize=fs)
  # loop through step sizes
  for hi,h in enumerate(H):
    K = np.int(np.ceil(T / h))
    t = h*np.arange(K)

    st = time.time() 
    PT = np.zeros((n*m,n))
    def dP(t,P,u):
      PA = np.dot(P,A)
      ATP = np.vstack([np.dot(A.T,P[i*n:(i+1)*n]) for i in range(m)])
      PBRiBTP = - np.vstack([np.dot(P[i*n:(i+1)*n],
                               sum([np.dot(B_[j],
                                           np.dot(RI[j][j],
                                                  np.dot(B_[j].T,
                                                         P[j*n:(j+1)*n]))) 
                                    for j in range(m)])) 
                        for i in range(m)])
      return PA + ATP + Q + PBRiBTP

    dP_ = lambda t,P,u : (np.dot(A.T,P.T).T + np.dot(P,A) + Q
           - np.vstack([np.dot(P[i*n:(i+1)*n],
                               sum([np.dot(B_[j],
                                           np.dot(RI[j][j],
                                                  np.dot(B_[j].T,
                                                         P[j*n:(j+1)*n]))) 
                                    for j in range(m)])) 
                        for i in range(m)]))

    P = euler(h, K, dP, PT, np.zeros((K,0)))[::-1]
    P_ = euler(h, K, dP_, PT, np.zeros((K,0)))[::-1]
    assert not np.isnan(P[0]).any() # adjoint diverged !
    print '|P - P_| = %0.2f' % np.sqrt(np.sum((np.vstack(P) - np.vstack(P_))**2))

    dxLQG = lambda t,x,P : (np.dot(x,A.T) 
              + np.dot(np.hstack([-np.dot(np.dot(np.dot(x,
                                                        P[i*n:(i+1)*n]),
                                                 B_[i]),
                                          RI[i][i]) 
                                  for i in range(m)]),B.T))
    xLQG = np.asarray(euler(h, K, dxLQG, x0, P))
    uLQG = np.asarray([np.hstack([-np.dot(np.dot(np.dot(x,P[i*n:(i+1)*n]),B_[i]),RI[i][i]) for i in range(m)]) for x,P in zip(xLQG,P)])
    dx = lambda t,x,u : (np.dot(x,A.T) + np.dot(u,B.T))
    Xlqg = np.array( euler(h, K, dx, x0, uLQG) )
    print '|X - x| = %0.2f' % np.sqrt(np.sum((Xlqg - xLQG)**2))

    J = lambda u : np.asarray([.5*h*np.cumsum( np.sum(np.dot(xLQG,Q_[i])*xLQG,axis=1) 
                    + np.sum(np.dot(u,Rj[i])*u,axis=1) ) for i in range(m)]).T

    JLQG = np.vstack([np.zeros(m),J(uLQG)[:-1]])

    U = dict(lqg=uLQG); X = dict(x=xLQG)

    print '%0.2fsec for %6s, J(T) = %s' % ((time.time() - st), 'lqg', JLQG[-1])

    plt.figure(1); plt.clf()
    
    ax1 = plt.subplot(3,1,1)
    ax1.grid('on')
    ax1.plot(t,xLQG,'k',alpha=.5,lw=6,label='LQG')
    ax1.set_ylabel('$x$')
    #ax.legend(loc=8,ncol=4)

    ax2 = plt.subplot(3,1,2)
    ax2.grid('on')
    ax2.plot(t,uLQG,'k',alpha=.5,lw=6,label='LQG')
    ax2.set_ylabel('$u$')
    #ax.legend(loc=8,ncol=4)

    ax3 = plt.subplot(3,1,3)
    ax3.grid('on')
    ax3.plot(t,JLQG,'k',alpha=.5,lw=6,label='$J_{lqg}$')
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('$J$')

    # numerical optimal control
    #h = 2e-1
    K = np.int(np.ceil(T / h))
    t = h*np.arange(K)

    F = lambda t,x,u : np.hstack(( np.dot(x[:-m],A.T) + np.dot(u,B.T),
               [.5*(np.sum(np.dot(Q_[i],x[:-m])*x[:-m]) 
               + np.sum(np.dot(u,Rj[i])*u)) for i in range(m)] ))
    J = lambda x : x[-m:]

    DxF = lambda t,x,u : np.hstack(( np.vstack(( A, np.dot(x[:-m],Q.T).reshape(m,n) )),
                                     np.zeros((n+m,m)) ))
    DuF = lambda t,x,u : np.vstack(( B, np.dot(u,R.T).reshape(m,l) ))
    DxJ = lambda x : np.hstack(( np.zeros((m,n)), np.identity(m) ))

    dg = DG(L,F,J,DxF,DuF,DxJ)

    ls = 'g'; lw = 2; fs = 'nlp-lq'
    st = time.time() 
    z0 = np.hstack((x0,np.zeros(m)))
    u0 = np.zeros((K,l))
    #u0 = uLQG + np.random.randn(K,l)*1e+1
    u0 = uLQG

    # solve nlp
    u = dg.nlp(h,K,z0,u0,jmax=maxiter)
    # simulate at optimum
    z = dg.state(h,K,z0,u)
    # store result
    U[fs] = u; X[fs] = z[:,:-m]

    ax1.plot(t,z[:,:-m],ls,lw=lw,label=fs)
    ax2.plot(t,u,ls,lw=lw,label=fs)
    ax3.plot(t,z[:,-m:],ls,lw=lw,label='$J_{'+fs+'}$')
    print '%0.2fsec for %6s, J(T) = %s, j = %d' % ((time.time() - st), fs, z[-1,-m:],dg.res[-1])

    ls = 'b--'; lw = 4; fs = 'nlp0'
    st = time.time() 
    z0 = np.hstack((x0,np.zeros(m)))
    u0 = np.zeros((K,l))

    # solve nlp
    u = dg.nlp(h,K,z0,u0,jmax=maxiter)
    # simulate at optimum
    z = dg.state(h,K,z0,u)
    # store result
    U[fs] = u; X[fs] = z[:,:-m]

    ax1.plot(t,z[:,:-m],ls,lw=lw,label=fs)
    ax2.plot(t,u,ls,lw=lw,label=fs)
    ax3.plot(t,z[:,-m:],ls,lw=lw,label='$J_{'+fs+'}$')
    print '%0.2fsec for %6s, J(T) = %s, j = %d' % ((time.time() - st), fs, z[-1,-m:],dg.res[-1])

    #ax3.legend(loc=8,ncol=2)
    sc = 1.25*np.array([-1,1])
    ax1.set_ylim(np.abs(xLQG).max()*sc)
    ax2.set_ylim(np.abs(uLQG).max()*sc)
    ax3.set_ylim((0.,sc[1]*JLQG[-1].max()))

    DJ = lambda u_ : -np.sum(dg.DJ_(u_),axis=0)
    HJ = lambda u_ : jac(DJ,u_.reshape(K*l,order='F'),d=1e-1)

    DJs = ''
    for us in U:
      DJs += '|DJ_%s| = %0.4f; ' % (us,np.linalg.norm(dg.DJ_(U[us])))
    print DJs[:-2]

    if 0:#K < 25:
      HJs = ''
      for us in U:
        HJs += '|HJ_%s| = %0.4f; ' % (us,np.linalg.eigvals(HJ(U[us])).min())
      print HJs[:-2]

    e[hi] = np.sqrt( np.sum( (u - uLQG)**2 ) ) / np.sqrt( np.sum( uLQG**2 ) )

    ax0.loglog(K,e[hi],'b.',ms=ms,zorder=2,basex=10,basey=10)
    fig0.canvas.draw()

  ax0.loglog(np.ceil(T / H),e,'b',lw=lw,zorder=0,basex=10,basey=10)
  fig0.canvas.draw()


  for sfx in sfxs:
    if not os.path.exists(sfx):
      os.mkdir(sfx)
    f = os.path.join(sfx,fi+'.'+sfx)
    print 'saving %s' % f
    fig0.savefig(f)

  return dg

if __name__ == "__main__":

  import sys
  args = sys.argv

  flag = '--seed'
  if flag in args:
    seed = int(args[args.index(flag)+1])
  else:
    seed = np.int(10000*np.random.rand())

  flag = '--maxi'
  if flag in args:
    maxiter = int(args[args.index(flag)+1])
    print 'maxiter = %s' % maxiter

  dbg = False
  flag = '--dbg'
  if flag in args:
    dbg = True
  print 'dbg = %s' % dbg

  if '--circ' in args:
    if not '--seed' in args:
      seed = 9003
    if not '--maxi' in args:
      maxiter = 8*400
    #do_circ(seed, maxiter, dbg, sfxs=[])
    do_circ(seed, maxiter, dbg)

  if '--nlp' in args:
    if not '--seed' in args:
      seed = 2160
    if not '--maxi' in args:
      maxiter = 1000
    print args,seed
    #dg = do_nlp(seed, maxiter, dbg, sfxs=[])
    dg = do_nlp(seed, maxiter, dbg)
