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
  
  coxmodel <- cph(Surv(time,status)~hits+nv0+nv1+nv2+nv3+nv4+nv5+nv6+
                    nv7+nv8+nv9+nv10+nv11+nv12+nv13+nv14+nv15+nv16+
                    nv17+nv18+nv19+nv20+nv21+nv22+nv23+nv24+nv25+nv26+
                    nv27+nv28+nv29+nv30+nv31+nv32+nv33+nv34+nv35+nv36+
                    nv37+nv38+nv39+weight+user1degree+user2degree+
                    user1PG+user2PG+CN+user1cc+user2cc+jaccard+adamicadar+prefattachment,data=df_train,surv=TRUE)
  ttt <- c(1,2,3,4,5)
  pf<-predictSurvProb(coxmodel,newdata=df_test,times=ttt)
  k<- paste("fb",toString(i),"survProb",toString(j),".csv",sep="")
  write.table(pf,file=k,sep='\t',na="NA",dec=".",row.names=FALSE,col.names=TRUE)
  ApparrentCindex <- pec::cindex(pf,formula=Surv(time,status)~hits+nv0+nv1+nv2+nv3+nv4+nv5+nv6+
                                   nv7+nv8+nv9+nv10+nv11+nv12+nv13+nv14+nv15+nv16+nv17+nv18+nv19+
                                   nv20+nv21+nv22+nv23+nv24+nv25+nv26+nv27+nv28+nv29+nv30+nv31+nv32+
                                   nv33+nv34+nv35+nv36+nv37+nv38+nv39+weight+user1degree+user2degree+
                                   user1PG+user2PG+CN+user1cc+user2cc,data=df_test,eval.times=ttt)
  k<- paste("fb",toString(i),"cindex",toString(j),".csv",sep="")
  write.table(ApparrentCindex$AppCindex,file=k,sep='\t',na="NA",dec=".",row.names=FALSE,col.names=TRUE)
  
  
}
}
