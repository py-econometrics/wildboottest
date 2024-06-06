
****************************************************
*  Code for Table 2 of Kline and Santos (2011)     *
*  This file computes Analytical Wald and LM tests *
*  along with Score bootstrapped tests.   	   *
*  Some other results not reported in the paper    *
*  are also included.				   *	
****************************************************

version 11.1
set seed 12345

**mata subroutine to compute recentered clustered outer products**
**inputs a matrix S of scores and a cluster summing matrix L**
**outputs clustered outer product matrix**
**note this program assumes a balanced design**
**and proper sort order**
cap mata: mata drop clustOPG()
mata: 
real matrix clustOPG(real matrix S,real matrix L){
real scalar N
real matrix Scent, Scentsum, OPGcent

N=rows(S)
Scent=S - J(N,1,1)*mean(S)
Scentsum=L'*Scent
OPGcent=Scentsum'*Scentsum
return(OPGcent)
}
end



/* cap prog drop sim_probit
prog def sim_probit, rclass
syntax [, c(integer 5) m(integer 50) f(integer 0) r(integer 199)] */
*Notes on syntax: 
*c - # of clusters
*m - # of obs per cluster
*f - controls whether to use mixture regressor
*r - # of bootstrap reps per simulation

local c 10
local m 100
local f 0
local r 1000

**Generate data**
drop _all
set obs `c'  //generate c clusters

gen x=invnorm(uniform()) //covariate
gen mix=uniform()>.1 //contamination probability

if `f'==1{
	gen D=(mix*invnorm(uniform()) + (1-mix)*(2+3*invnorm(uniform())))/4 //regressor of interest
}
else{
	gen D=(invnorm(uniform()))/4 //regressor of interest
}

gen v=invnorm(uniform())


gen c=_n	//generate cluster id

expand `m'	//make m obs per cluster

replace x=(x+invnorm(uniform()))/4  //add some within cluster variation in x

gen y=x+D+v/sqrt(2)+invnorm(uniform())/sqrt(2)  //form outcome
replace y=y>0
sort c

**Restricted**
constraint 1 D=1
glm y x D, constraint(1) fam(bin) link(probit)
predict pr
predict xbr, xb
gen phir=normalden(xbr)
gen er=(y-pr)*phir/(pr*(1-pr)) //quasi-residual
gen wr=phir^2/(pr*(1-pr)) //weights for hessian
gen one=1

mat accum Hr=one x D [iw=wr], nocons //form hessian */


**Unrestricted**
probit y x D, cluster(c)
global b=_b[D]
local stdof=sqrt((`m'*`c'-3)/(`m'*`c'-1)) //inverse stata DOF correction
global W=((_b[D]-1)/(_se[D]*`stdof'))^2 //save analytical Wald

predict pu
predict xbu, xb
gen phiu=normalden(xbu)
gen eu=(y-pu)*phiu/(pu*(1-pu)) //quasi-residual
gen wu=phiu^2/(pu*(1-pu)) //weights for hessian

global dfk=(`c'/(`c'-1))
mata: L=I(`c')#J(`m',1,1) //matrix for summing across clusters
mata: X=st_data(.,("one", "x", "D"))
mata: C=(0\0\1)		//constraint matrix -- third coefficient is restricted

mata: Hr=st_matrix("Hr")
mata: Hrinv=invsym(Hr)
mata: Ar=C'*Hrinv		//form A_n


**Prepare influence function wald
mat accum Hu=one x D [iw=wu], nocons //form hessian
mata: Hu=st_matrix("Hu")
mata: Huinv=invsym(Hu)
mata: Au=C'*Huinv
mata: eu=st_data(.,"eu")
mata: OPGucent=clustOPG(X:*eu,L)
mata: Vinvu=invsym(Au*OPGucent*$dfk*Au') //compute inverse of variance of influence function


save "sim_sample.dta", replace

mata: S = X:*eu

**Bootstrap**
gen u=.
gen ystar=.
gen ernew=.
gen eunew=.
gen pos=.


mat Ts=J(1,6,.) //store bootstrap statistics

by c: replace u=uniform()
by c: replace pos=u[1]<.5  //cluster level rademacher indicator
gen radem = 2*pos - 1

qui replace ernew=(2*pos-1)*er  //weighted residual
qui replace eunew=(2*pos-1)*eu  //weighted residual

*Score bootstrap LM -- Rademacher
mata: er=st_data(.,"ernew")
mata: Sr=colsum(X:*er)
mata: OPGrcent=clustOPG(X:*er,L)
mata: lm=Sr*Ar'*invsym(Ar*(OPGrcent)*$dfk*Ar')*Ar*Sr'
mata: st_numscalar("lm",lm)
mat Ts[1,1]=lm	     //save bootstrap LM 


*Score Wald -- Rademacher
mata: eu=st_data(.,"eunew")
mata: Su=colsum(X:*eu)
mata: OPGucent=clustOPG(X:*eu,L)
mata: wald=Su*Au'*invsym(Au*OPGucent*$dfk*Au')*Au*Su'
mata: st_numscalar("wald",wald)
mat Ts[1,2]=wald	     //save bootstrap Wald

*Score bootstrap Wald combining restricted and unrestricted scores -- Rademacher
mata: wald_r=Sr*Ar'*invsym(Au*OPGucent*$dfk*Au')*Ar*Sr'
mata: st_numscalar("wald_r",wald)
mat Ts[1,3]=wald	     //save bootstrap Wald


