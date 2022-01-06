libname card "D:\collectioncard";

proc import out=card.test7
datafile="F:\7.番茄学院开课啦\1.催收评分卡\trains"
dbms=excel replace;
getnames=yes;
mixed=yes;
scantext=yes;
usedate=yes;
scantime=yes;
run;
