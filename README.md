# NonconvexFW-MatrixCompletion

This is a Matlab package that implements the nonconvex Frank-Wolfe (FW) method and away-step Frank-Wolfe (AFW) method for solving matrix completion problem of the form:

$$\eqalign{
\min\limits_{X\in R^{m\times n}} &\frac12\sum\limits_{(i,j)\in\Omega}(X_{ij}-\bar X_{ij})^2 \\
\text{s.t.} & ||X||_* - \mu ||X||_F \leq \sigma,
}$$

where $\bar X$ is the observation and $\Omega$ is a collection of indices and $\sigma > 0$. The methods are compared with the [InFaceExtendedFW-MatrixCompletion](https://github.com/paulgrigas/InFaceExtendedFW-MatrixCompletion) solver hosted by Paul Grigas. 

A paper presenting more details on the implementations can be found here: 
> [Frank-Wolfe type methods for nonconvex
inequality-constrained problems](https://arxiv.org/pdf/2112.14404.pdf).

<br />

### First time setup

1. Clone the repository: git clone **--recurse-submodules** https://github.com/zengliaoyuan/NonconvexFW_MC.git
2. Install the mex file for the running: 
      - Set the current working path as "./NonconvexFW_MC/InFaceExtendedFW-MatrixCompletion/solver" in Matlab
      - Excute in the command window: mex project_obs_UV.c
3. Change the working path to "./NonconvexFW_MC" in Matlab


**Remark**: Our coding are done with Matlab 2017b for Windows. One has to change the formats of the path names involved in the package if using Matlab for other operating systems.


### Runcodes for comparing FW, AFW and the InFaceExtendedFW-MatrixCompletion 


**runcode_movLens**: Runcode for generating the figures in the numerical experiments of the paper<br />
**run_table**: Runcode for generating the table in the numerical experiments of the paper<br />



### Other Matlab source codes are:


**FW_nuc**: Main function that combines the implementations of the FW and AFW method<br />
**Lomat**: Subroutine of the Frank-Wolfe linear optimization oracle<br />
**awstep_nuc**: Subroutine of the away-step oracle<br />
**eigifp**: A solver computes the smallest (largest) eigenvalue of a generalized eigenvalue problem and is downloaded here: [eigifp](https://www.ms.uky.edu/~qye/software.html) <br />
**update_svd_asIF**: Rank-one SVD updation; this is a part of the 'update_svd' file in the InFaceExtendedFW-MatrixCompletion solver<br />
**savefig**: Subroutine for saving figures in the runcode_movLens

