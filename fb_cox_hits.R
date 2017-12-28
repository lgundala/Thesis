library('survival')
library('pec')
library(prodlim)
library(rms)
for(i in 0:9){
  for(j in 0:4){
    print(i)
    print(j)
    pf<- paste("fb",toString(i),"train",toString(j),".csv",sep="")
    df_train = read.table(pf,sep='\t',header=TRUE)
    pf<- paste("fb",toString(i),"test",toString(j),".csv",sep="")
    df_test = read.table(pf,sep='\t',header=TRUE)
    
    coxmodel <- cph(Surv(time,status)~hits,data=df_train,surv=TRUE)
    ttt <- c(1,2,3,4,5)
    pf<-predictSurvProb(coxmodel,newdata=df_test,times=ttt)
    k<- paste("fb",toString(i),"survProbh",toString(j),".csv",sep="")
    write.table(pf,file=k,sep='\t',na="NA",dec=".",row.names=FALSE,col.names=TRUE)
    ApparrentCindex <- pec::cindex(pf,formula=Surv(time,status)~hits,data=df_test,eval.times=ttt)
    k<- paste("fb",toString(i),"cindexh",toString(j),".csv",sep="")
    write.table(ApparrentCindex$AppCindex,file=k,sep='\t',na="NA",dec=".",row.names=FALSE,col.names=TRUE)
    
    
  }
}
