library(dplyr)
library(ggplot2)
library(ggthemes)
library(corrplot)
input<-read.csv("E:/R/ibm.csv")
summary(input)
str(input)
names(input)
dim(input)
num_input<-input[,c(1,4,6,7,10,11,13,14,15,17,19,24,25,26,28:35)]
num_attri=as.numeric(input$Attrition)
num_data=cbind(num_input,num_attri)
str(num_data)
M<-cor(num_data)
corrplot(M,method="number",insig="blank")
k=0
for (i in 1:23)
{
for (r in 1:23)
  {
    if (M[i,r] > 0.70 & i!=r)
    {
      k=k+1
    }
  }
}
print(k/2)
#Overtime vs Attrition
a<-ggplot(input,aes(OverTime,fill=Attrition)) + geom_histogram(stat="count")
print(a)
tapply(as.numeric(input$Attrition),input$OverTime,mean)
#Business Travel vs Attrition
B<-ggplot(input,aes(BusinessTravel,fill=Attrition))+geom_histogram(stat="count")
print(B)
tapply(as.numeric(input$Attrition),input$BusinessTravel,mean)
#Distance from Home vs Attrition
C<-ggplot(input,aes(DistanceFromHome,fill=Attrition)) + geom_histogram(stat="count")
print(C)
D<-tapply(as.numeric(input$Attrition),input$DistanceFromHome,mean)
print(which(D>1.3))

