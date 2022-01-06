data ben.train2;
set total_train_num;
run;



/*多重共线性诊断*/
	proc reg data=ben.train2 ;
	model target =woe: /collin vif selection=none;
	ods output parameterestimates=ben.reg_1 collindiag=ben.reg_2;
	quit;



/*以此建模为模板增加变量*/
data ben.train3;
set ben.train2(keep=
target
loan_no
woe_BP_L4M
woe_BP_L7M
woe_DPD_L4M
woe_DPD_L7M
woe_KPTP_RATE_SUM
woe_NOPAY_L3M
woe_PERIOD_PERCEN
woe_PTP_L1M
)
;
run;

proc logistic data=ben.train3   outest=ben.train_stat desc;
model  target(event='1')= woe: 
/ selection=s  sle=0.5 sls=0.5
;
output out =ben.model_pred  p = phat ;
run;

Ods Output ParameterEstimates=aa_1 ;
proc npar1way data=ben.model_pred  noprint;
      class target;
      var phat;
      output out=ben.ks_1(keep=_d_ p_ksa rename=(_d_=KS p_ksa=P_value));
run;

/***对系数转置****/
proc transpose  data=ben.Train_stat
out=ben.temp_mpt;
run;

/*******/
data ben.temp_mpt;
set ben.temp_mpt(where=(target is not null));
run;





data ben.temp_mpt1;
set  ben.temp_mpt;
ix=find(_Name_,'woe_');
if ix>0 then varname=upcase(substr(_Name_,5));
else varname=_Name_;
run;

proc transpose data=ben.temp_mpt1   out=ben.par_zh  prefix=p_;
     id varname;
     var target;
run;

proc sql;
     create table ben.benbenben as select a.*,b.*
     from ben.train2  a
     left join ben.par_zh  b
     on 1=1;
quit;


%macro pro_js();
     proc sql noprint;
           select count(distinct varname) into:count from ben.Temp_mpt1;
     quit;
     %put &count.;
     %do i=1 %to &count;
           data _null_;
           set ben.Temp_mpt1;
           if _n_=&i;
           call symputx("var",varname);
           run;
           %put &var.;

           data ben.benbenben;
           set  ben.benbenben;
           z_&var.=woe_&var.*p_&var. ;
           run;
     %end;

%mend;
%pro_js();




data ben.benbenben2;
     set ben.benbenben(drop = z_intercept woe_intercept  z__LNLIKE_ woe__LNLIKE_ );
a_sum=sum(of z_:);    
run;

proc sql noprint;
     select target   into:Intercept  from ben.Temp_mpt1 where varname="Intercept";
quit;
%put&intercept.;

/*再加上截距，并进行类概率值计算*/
data ben.benbenben3;
     set ben.benbenben2;
     odds=a_sum+&Intercept.;
     e_odds=exp(odds);
     phat=e_odds/(1+e_odds);
run;
/*输出ks*/
proc npar1way data=ben.benbenben3  noprint ;
     class target;
     var phat;
     output out=ben.ks_valid(keep=_d_ p_ksa rename=(_d_=KS p_ksa=P_value));
run;
/*转换标准评分*/
%macro score(pdo,PO,M);
data  ben.benbenben4;
set ben.benbenben3;
  %let   B= &pdo/log(2);
  %let   A=&PO+&B*log(&M);;
     cs=log(phat/(1-phat));
  score=ROUND(&A-&B*cs,1);
  %put &B;
    %put &A;
run;
%mend; 
%score(PO=500,pdo=50,M=1/20);

/*计算gini*/
proc freq data=ben.benbenben4  noprint ;
tables phat*target;
test  smdrc;
output out=ben.gini_valid(keep=_SMDRC_ ) smdrc;
run;


  proc sql noprint;
   create table  ben.temp2 as
   select * from ben.Temp_mpt1
   where varname not in ('Intercept','_LNLIKE_') ;
   quit;

proc sql noprint;
create table ben.bmw  as
select distinct a.* ,b.target  from   ben.woe   as  a
inner join  ben.temp_mpt1    as  b
on a.var=b.varname
order by var;
quit;



/*****进行评分卡标志分数的计算*********/
%macro score1(PO,pdo,M);
data ben.woe_score;
set ben.bmw;
  %let b=%sysevalf(&pdo./%sysfunc(log(2)));
  %let A=%sysevalf(&PO+&b*%sysfunc(log(&M)));
   woe_score=-woe*target*&b;
total_score=-woe*target*&b+(&A.-&b*(-1.7611))/8;
run;
%put &b;
%put &A;
%mend; 
%score1(PO=500,pdo=50,M=1/20);


