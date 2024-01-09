# NonconvexFW-MatrixCompletion

This is a Matlab package that implements the nonconvex Frank-Wolfe (FW) method and away-step Frank-Wolfe (AFW) method for solving matrix completion problem of the form:

$$\eqalign{
\min\limits_{X\in R^{m\times n}} &\frac12\sum\limits_{(i,j)\in\Omega}(X_{ij}-\bar X_{ij})^2 \\
\text{s.t.} & ||X||_* - \mu ||X||_F \leq \sigma,
}$$

where $\bar X$ is the observation and $\Omega$ is a collection of indices and $\sigma > 0$. The methods are compared with the [InFaceExtendedFW-MatrixCompletion](https://github.com/paulgrigas/InFaceExtendedFW-MatrixCompletion) solver hosted by Paul Grigas. 

A paper presenting more details on the implementations can be found here: 
> [Frank-Wolfe-type methods for a class of nonconvex inequality-constrained problems](https://arxiv.org/pdf/2112.14404.pdf).

<br />

### First time setup

1. Clone the repository: git clone **--recurse-submodules** https://github.com/zengliaoyuan/NonconvexFW_MC.git
2. Install the mex file for the running: 
      - Set the current working path as "./NonconvexFW_MC/InFaceExtendedFW-MatrixCompletion/solver" in Matlab
      - Excute in the command window: mex project_obs_UV.c
3. Change the working path to "./NonconvexFW_MC" in Matlab


**Remark**: Our coding are done with Matlab 2022b for Windows. One has to change the formats of the path names involved in the package if using Matlab for other operating systems.


### Runcodes for comparing FW, AFW and the InFaceExtendedFW-MatrixCompletion with datasets MovieLens10M, MovieLens20M, MovieLens32M and Netflix Prize


**run_all_datasets**: Runcode for generating the table and figures in the numerical experiments of the paper<br />
**runcode_movLens_10M**, **runcode_movLens_20M**, **runcode_movLens_32M** and**runcode_netflix_100M**: Subroutines related to MovieLens10M,  MovieLens20M, MovieLens32M and Netflix Prize accordingly; used in 'run_all_datasets'

 




### Other Matlab source codes are:


**FW_nuc**: Main function that combines the implementations of the FW and AFW method<br />
**Lomat**: Subroutine of the Frank-Wolfe linear optimization oracle<br />
**awstep_nuc**: Subroutine of the away-step oracle<br />
**eigifp**: A solver computes the smallest (largest) eigenvalue of a generalized eigenvalue problem and is downloaded here: [eigifp](https://www.ms.uky.edu/~qye/software.html) <br />
**update_svd_asIF**: Rank-one SVD updation; this is a part of the 'update_svd' file in the InFaceExtendedFW-MatrixCompletion solver<br />
**find_delta_asIF**: A cross-validation routine for learning $\delta$ of the nuclear norm constraint in the InFaceExtendedFW-MatrixCompletion. Here take $\sigma:=\delta$ in $||X|_*-\mu||X||_F\leq \sigma$ for all datasets<br />
**savefig**: Subroutine for saving figures in 'run_all_datasets'

