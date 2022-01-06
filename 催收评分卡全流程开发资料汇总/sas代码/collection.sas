libname ben'F:\7.番茄学院开课啦\1.催收评分卡';
proc import out=ben.train
datafile="F:\7.番茄学院开课啦\1.催收评分卡\trains"
dbms=excel replace;
getnames=yes;
mixed=yes;
scantext=yes;
usedate=yes;
scantime=yes;
run;


proc import out=ben.group2
datafile="F:\7.番茄学院开课啦\1.催收评分卡\演示数据\fenzu0429.xls"
dbms=excel replace;
getnames=yes;
mixed=yes;
scantext=yes;
usedate=yes;
scantime=yes;
run;


data  ben.group_num2;
set ben.group2;
if varGroup^=.  and  varType="N" ;
run;
proc sql noprint;
create table ben.num2  as 
select distinct varname  from  ben.group_num2;
quit;



%macro split_number();

data _null_;
	set ben.num2 NOBS=last;
	call symput('N_VAR_NUM',last); 
	stop;
run; 

data _null_;
	set  ben.num2;
	%do i=1 %to &N_VAR_NUM;
	if _N_=&i then do;
			call symput("TRAIN_VAR_NUM_&i",compress('FCT_'||varname)||'(keep=target loan_no  nresp resp '||compress(varname)||')'); 
	end;
	%end;



%macro train_var_NUM;
	%do i=1 %to &N_VAR_NUM;   
		&&TRAIN_VAR_NUM_&i
	%end;
%mend;
run; 


data %train_var_NUM;
	set ben.train;
	if target=1 then do;
		nresp = 1;
		resp = 0;
	END;
	ELSE DO;
		nresp = 0;
		resp = 1;
	END; 
run;

proc sql noprint;
		drop table train_total;
quit;



%do i=1 %to &N_VAR_NUM;
	data _null_;
		set ben.num2;
		if _N_=&i then do;
			call symput('SPLIT_VAR_NUM',varName);
		end;
	run; 

proc sql noprint;
		select count(*) into:group
		from ben.group_num2  where varName="&SPLIT_VAR_NUM";
		%put &group.;
		%PUT &SPLIT_VAR_NUM;
	quit;

proc sql;
		drop table CDE_&SPLIT_VAR_NUM.;
quit;

data grp_&split_var_num(keep=varname  nSplitVal);
	set ben.group2;
	if varGroup^=.    ;
	if varname="&SPLIT_VAR_num";
	if varType="N" ;
run;


	%if %eval(&group)>0 %then %do;
		%do j=1 %to (&group-1);  
		data _null_;
			set GRP_&SPLIT_VAR_num;
			%if &j=1 %then %do; 
			if _n_=&j then do;
				call symput("X",nSplitVal);

			end;
			%end;

			%else %if &j>1 %then %do;
			if _n_=%eval(&j-1) then do;
				call symput("X_1",nSplitVal);
			end;
			if _n_=&j then do;
				call symput("X_2",nSplitVal);
			end; 
			%end;
		  run; 

	    	proc sql noprint;
			create table group_tmp as
			select loan_no,
				&SPLIT_VAR_num,
				target,nresp,resp
				
				,&j as grp
			from fct_&SPLIT_VAR_num
			%if &j=1 %then %do; 
			where &SPLIT_VAR_NUM<&X 
			%end;
			%else %if &j>1 %then %do; 
			where &X_1<=&SPLIT_VAR_num<&X_2 
			%end;
			;
		   quit; 

			proc append base=CDE_&SPLIT_VAR_NUM. data=group_tmp;
			run;  
			**&j=&group 在上述处理后再多进行两步处理;
			%if &j=(&group-1) %then %do;

			data _null_;
			set GRP_&SPLIT_VAR_NUM; 
			if _n_=&j  then do;
				call symput("X",nSplitVal);
			end; 
		    run; 

			proc sql;
			create table group_tmp  as
			select loan_no,
				&SPLIT_VAR_num,
				target,nresp,resp
				
				,&group as grp
			from fct_&SPLIT_VAR_num
			where &SPLIT_VAR_num>=&X
			;
		quit; 
			proc append base=CDE_&SPLIT_VAR_NUM  data=group_tmp ;
		run;
			%END;			
			run;

		%end;

	 %end;

   data CDE_&SPLIT_VAR_NUM;
   set  CDE_&SPLIT_VAR_NUM;
   if &SPLIT_VAR_NUM=. then do;
   grp=1;
   end;
   run;


	data woe_&SPLIT_VAR_num;
			length Var $40. VAR_LST 8 Var_GRP $40. TOT_IV sum_resp sum_nresp _freq_ badrate TOTALGOOD TOTALBAD PERCGOOD PERCBAD ODDS WOE IV 8.; 
			stop;
		run;

	proc summary data=cde_&SPLIT_VAR_num(keep=grp resp nresp) missing;
			class grp;
			var resp nresp;
			output out=fct2 sum=sum_resp sum_nresp;
		run;

	data fct3;
		retain grp;
		label sum_resp = "GOOD" sum_nresp = "BAD" _freq_ = "G+B" badrate = "BAD RATE(%)";
		if _n_=1 then set fct2(keep=sum_resp sum_nresp rename=(sum_resp=TOTALGOOD sum_nresp=TOTALBAD));                                   
		set fct2(where = (_type_ = 1));     
		if sum_resp = 0 then PERCGOOD=0.5/TOTALGOOD;
		else PERCGOOD=sum_resp/TOTALGOOD;
		if sum_nresp = 0 then PERCBAD=0.5/TOTALBAD;
		else PERCBAD=sum_nresp/TOTALBAD;
		BADRATE = sum_nresp / _freq_ ;
		ODDS=PERCGOOD/PERCBAD;
		WOE = log(ODDS);
		IV= (PERCGOOD-PERCBAD)*WOE;  
		format badrate 10.4;
		drop _TYPE_;
	run;

	proc sql noprint;
		select sum(IV) into: TOT_IV
		FROM FCT3;
	QUIT;

	proc sql noprint;
		select sum(IV) into: TOT_IV
		FROM FCT3;
	QUIT;

	DATA WOE_&SPLIT_VAR_NUM(DROP=&SPLIT_VAR_NUM);
		LENGTH Var $40 GRP  TOT_IV 8;
		SET FCT3;
		VAR="&SPLIT_VAR_NUM"; 
		TOT_IV=&TOT_IV;

	RUN;

	proc sql;
		create table T_&SPLIT_VAR_NUM
		as
		select a.loan_no,
			a.&SPLIT_VAR_NUM
			,a.target
			,b.woe
		from cde_&SPLIT_VAR_NUM as a 
		inner join woe_&SPLIT_VAR_NUM as b 
		on a.grp=b.grp
		;
	quit;	

	data T_&SPLIT_VAR_NUM(rename=(woe=woe_&SPLIT_VAR_NUM));
	set T_&SPLIT_VAR_NUM;
	run;

	proc sort data=T_&SPLIT_VAR_NUM;by loan_no;run;

   %if &i=1 %then %do; 
   data total_train_num;
   set T_&SPLIT_VAR_NUM;
   run;
   data woe_num;
   set WOE_&SPLIT_VAR_NUM ;
   run;
   %end;
   %else %do;
   data  total_train_num;
   merge total_train_num T_&SPLIT_VAR_NUM;
   by loan_no;
   run;
   data woe_num;
   set woe_num WOE_&SPLIT_VAR_NUM ;
   run;
   %end;
  

%end;

%mend;
%split_number;





  
