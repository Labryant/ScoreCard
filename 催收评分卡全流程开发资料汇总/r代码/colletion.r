################################# BEGIN ###################################################################################
#一、导入数据
	library(readr)
	O_data<-read.csv('F:/trains.csv',header = TRUE)
	O_data$bad_ind<- as.factor(O_data$bad_ind)

#二、数据清洗
	#2.1.检查数据类型及数据缺失
		Vname<- NULL
		VCla<- NULL
		Vna<- NULL
		for (i in (1:length(O_data))){
		  vnm<- names(O_data)[i]
		  cla<- class(O_data[,i])
		  vna<- sum(is.na(O_data[,i]))
		  Vname<- c(Vname,vnm)
		  VCla<- c(VCla,cla)
		  Vna<- c(Vna,vna)
		}
		V_Cla<- cbind(Vname,VCla,Vna)
		write.csv(V_Cla,file = "C:/Users/Administrator/Desktop/番茄数据/data/01_数据类型.csv",row.names = TRUE)
	#若存在数据缺失和数据类型不正确现象，在EXCEL中手工修改后重新导入数据;	
	
	#2.2 检查数据分布
	#通过直方图查看连续变量的分布，检查分布是否正常;
		hist(O_data$tot_income,col="red",main="月均收入")	
	#通过直方图查看分类变量的分布，检查分布是否正常;
		plot(O_data$bankruptcy_ind,col="red",main="是否破产过")
	#若存在某变量的数据分布存在异常，检查取数逻辑看是否正确。
	
#三、计算IV值
	#3.1 加载WOE包
		library(usethis)
		library(devtools)
		library(pcaPP)
		library(woe)
	#3.2 计算IV值
		V_name<- NULL
		I_v<- NULL
		for (i in 2:length(O_data)){
			vname<- names(O_data)[i]
			cla<-class(O_data[,i])
			we<- woe(Data=O_data
					,names(O_data)[i]
					,ifelse(cla=="factor",FALSE,TRUE)
					,"bad_ind"
					,10
					,Bad=0
					,Good=1
					)
			iv<- sum(ifelse(we$IV=="NaN",0,ifelse(we$IV=="Inf",0,we$IV)))
			V_name<- c(V_name,vname)
			I_v<- c(I_v,iv)
		}
		V_iv<- data.frame(V_name,I_v)	
	#3.3 查看计算结果
		View(V_iv)
	#3.4 将计算结果保存成csv文件
		write.csv(V_iv,file = "C:/Users/Administrator/Desktop/番茄数据/data/02_iv值计算.csv",row.names = TRUE)
	
#四、变量聚类;
	#4.1 将IV大于0.02的连续变量筛选出来
		sel<- c(
		  "LOAN_AMOUNT",
		  "PERIOD_PERCEN",
		  "Inbound_Call",
		  "DPD_TIME_L1M",
		  "DPD_TIME_L2M",
		  "DPD_TIME_L3M",
		  "DPD_TIME_L4M",
		  "DPD_TIME_L5M",
		  "DPD_TIME_L6M",
		  "DPD_TIME_L7M",
		  "DPD_TIME_L8M",
		  "DPD_TIME_L9M",
		  "DPD_TIME_L10M",
		  "DPD_TIME_L11M",
		  "DPD_TIME_L12M",
		  "DPD_TIME_SUM",
		  "KPTP_L1M",
		  "KPTP_L2M",
		  "KPTP_L3M",
		  "KPTP_L4M",
		  "KPTP_L5M",
		  "KPTP_L6M",
		  "KPTP_L7M",
		  "KPTP_L8M",
		  "KPTP_L9M",
		  "KPTP_L10M",
		  "KPTP_L11M",
		  "KPTP_L12M",
		  "KPTP_SUM",
		  "BP_L1M",
		  "BP_L2M",
		  "BP_L3M",
		  "BP_L4M",
		  "BP_L5M",
		  "BP_L6M",
		  "BP_L7M",
		  "BP_L8M",
		  "BP_L9M",
		  "BP_L10M",
		  "BP_L11M",
		  "BP_L12M",
		  "BP_SUM",
		  "PTP_L1M",
		  "PTP_L2M",
		  "PTP_L3M",
		  "PTP_L4M",
		  "PTP_L5M",
		  "PTP_L6M",
		  "PTP_L7M",
		  "PTP_L8M",
		  "PTP_L9M",
		  "PTP_L10M",
		  "PTP_L11M",
		  "PTP_L12M",
		  "PTP_SUM",
		  "KPTP_RATE_L1M",
		  "KPTP_RATE_L2M",
		  "KPTP_RATE_L3M",
		  "KPTP_RATE_L4M",
		  "KPTP_RATE_L5M",
		  "KPTP_RATE_L6M",
		  "KPTP_RATE_L7M",
		  "KPTP_RATE_L8M",
		  "KPTP_RATE_L9M",
		  "KPTP_RATE_L10M",
		  "KPTP_RATE_L11M",
		  "KPTP_RATE_L12M",
		  "KPTP_RATE_SUM",
		  "DPD_L1M",
		  "DPD_L2M",
		  "DPD_L3M",
		  "DPD_L4M",
		  "DPD_L5M",
		  "DPD_L6M",
		  "DPD_L7M",
		  "DPD_L8M",
		  "DPD_L9M",
		  "DPD_L10M",
		  "DPD_L11M",
		  "DPD_L12M",
		  "DPD_SUM",
		  "DPD_L3M_SUM",
		  "DPD_L6M_SUM",
		  "DPD_L3M_MAX",
		  "DPD_L6M_MAX",
		  "DPD5_TIME",
		  "NOPAY_L1M",
		  "NOPAY_L2M",
		  "NOPAY_L3M",
		  "NOPAY_L4M",
		  "NOPAY_L5M",
		  "NOPAY_L6M",
		  "NOPAY_L7M",
		  "NOPAY_L8M",
		  "NOPAY_L9M",
		  "NOPAY_L10M",
		  "NOPAY_L11M",
		  "NOPAY_L12M",
		  "NOPAY_SUM",
		  "NOPAY_MAX"
				)
		Con_data<- O_data[,sel]
	#4.2 根据变量间的相似性做树形图
		library(ClustOfVar)
		tree<- hclustvar(Con_data)
		plot(tree)
	#4.3 检查树形图的稳定性
		stability(tree,B=10)
	#4.4 决定聚为10类
		part<- cutreevar(tree,10,matsim=T)
		part$sim
	#4.6相关性强的变量间，仅保留1个就好。
	
#五、变量分箱
	#5.1 加载SQL环境
		library(proto)
		library(gsubfn)
		library(RSQLite)
		library(sqldf)
		library(sqldf)
	#5.2 变量手工分箱

#六、变量WOE编码
	#6.1 将数据分成训练集和测试集
		accepts<- O_data
		set.seed(10)
		select<- sample(1:nrow(accepts),length(accepts$bad_ind)*0.7)
		Cdata_train<- accepts[select,]
		dim(Cdata_train)
		Cdata_test<- accepts[-select,]
		dim(Cdata_test)
	#6.2 将数据集woe编码
		Cdta_train_woe<-sqldf('
				SELECT
				bad_ind
				,CASE 
					WHEN =0   THEN -0.47 
					WHEN <0.2 THEN -0.48 
					WHEN <0.3 THEN -0.12 
					WHEN <0.5 THEN 0.19 
					WHEN <1   THEN 0.61 
					WHEN >=1  THEN 1.03 
				END AS WOE_latest_6month_ovdue_loan_pct
				FROM Cdata_train
				')
		Cdta_test_woe<-sqldf('
				SELECT
				bad_ind
				,CASE 
					WHEN =0   THEN -0.47 
					WHEN <0.2 THEN -0.48 
					WHEN <0.3 THEN -0.12 
					WHEN <0.5 THEN 0.19 
					WHEN <1   THEN 0.61 
					WHEN >=1  THEN 1.03 
				END AS WOE_latest_6month_ovdue_loan_pct
				FROM Cdata_train
			')
	#6.2 检查是否有缺失值
		for (i in(1:dim(Cdta_train_woe)[2])){
			vn<- names(Cdta_train_woe)[i]
			nan<- sum(is.na(Cdta_train_woe[,i]))
			vnan<- c(vn,nan)
			print(vnan)
		}
		for (i in(1:dim(Cdta_test_woe)[2])){
			vn<- names(Cdta_test_woe)[i]
			nan<- sum(is.na(Cdta_test_woe[,i]))
			vnan<- c(vn,nan)
			print(vnan)
		}
		#若存在缺失值，回去检查代码6.1；
	#6.3 检查数据分布是否有异常
		summary(Cdta_train_woe)
		summary(Cdta_test_woe)
		#若存在分布异常，回去检查代码6.1；
	
#七、评分卡建模
	#7.1 逻辑回归模型
		#构建模型
		Lfit<- glm(bad_ind~.,Cdta_train_woe,family = "binomial")	
		#逐步回归法筛选变量
		lg_both<- step(Lfit,direction="both")
		#查看模型：lg_both
		summary(lg_both)	
		#在逐步回归法基础上手工筛选变量;
		LG_Lfit<- glm(bad_ind~WOE_latest_6month_ovdue_term_pct
							+WOE_ovdue_sts_days+WOE_h_max_ovdue_days
							+WOE_latest_6month_ovdue_term
							+WOE_m1_repay_cue
						,Cdta_train_woe
						,family = "binomial"
					)
		#查看模型：LG_Lfit
		summary(LG_Lfit)
		
	#7.2 转化成0-1000的评分
		pre_train<- predict(LG_Lfit,Cdta_train_woe)
		length(pre_train)
		summary(pre_train)
		pre_test<- predict(LG_Lfit,Cdta_test_woe)
		length(pre_test)
		summary(pre_test)
		avt<- mean(pre_train)
		#转化公式
		sco<- 1000/(1+exp(pre_test -avt))
		summary(sco)
	
#八、模型评估
	#8.1 KS计算：
		Cdta_test_woe$p<- predict(LG_Lfit,Cdta_test_woe,"response")
		TPR <- NULL
		FPR <- NULL
		for(i in seq(from=1,to=0,by=-0.1)){
		  #判为正类实际也为正类
		  TP1<- Cdta_test_woe[Cdta_test_woe$p >=i,]
		  TP2<- TP1[TP1$bad_ind ==1,]
		  TP<- dim(TP2)[1]
		  #判为正类实际为负类
		  FP1<- Cdta_test_woe[Cdta_test_woe$p >=i,]
		  FP2<- FP1[FP1$bad_ind ==0,]
		  FP<- dim(FP2)[1]
		  #判为负类实际为负类
		  TN1<- Cdta_test_woe[Cdta_test_woe$p <i,]
		  TN2<- TN1[TN1$bad_ind ==0,]
		  TN<- dim(TN2)[1]	  
		  #判为负类实际为正类
		  FN1<- Cdta_test_woe[Cdta_test_woe$p <i,]
		  FN2<- FN1[FN1$bad_ind ==1,]
		  FN<- dim(FN2)[1]
		  TPR <- c(TPR,TP/(TP+FN))
		  FPR <- c(FPR,FP/(FP+TN))
		}
		library(ggplot2)
		ggplot(data=NULL,mapping = aes(x=seq(0,1,0.1),y=TPR))+
		  geom_point()+
		  geom_smooth(se=FALSE,formula = y ~ splines::ns(x,10), method ='lm')+
		  geom_line(mapping = aes(x=seq(0,1,0.1),y=FPR),linetype=6)
		max(TPR-FPR)
	
	#8.2绘制ROC曲线
		#加载ROCR包
		library(gplots)
		library(ROCR)
		#就算模型预测的逾期概率
		Cdta_test_woe$p<- predict(Lfit_ya,Cdta_test_woe,"response")
		summary(Cdta_test_woe$p)
		Cdta_train_woe$p<- predict(Lfit_ya,Cdta_train_woe,"response")
		summary(Cdta_train_woe$p)
		#计算灵敏度和特异度
		pred_Te<- prediction(Cdta_test_woe$p,Cdta_test_woe$bad_ind)
		perf_Te<- performance(pred_Te,"tpr","fpr")
		pred_Tr<- prediction(Cdta_train_woe$p,Cdta_train_woe$bad_ind)
		perf_Tr<- performance(pred_Tr,"tpr","fpr")
		#画图
		plot(perf_Te,col='blue',lty=1)
		plot(perf_Tr,col='black',lty=2,add=TRUE)
		abline(0,1,lty=2,col='red')
		#计算AUC值并画图
		lr_m_auc<- round(as.numeric(performance(pred_Te,'auc')@y.values),3)
		lr_m_str<- paste("Mode_TEST_AUC:",lr_m_auc,sep="")
		legend(0.3,0.4,c(lr_m_str),2:8)
	
################################# END ################################################################################



