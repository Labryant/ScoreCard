libname card "D:\collectioncard";

proc import out=card.test7
datafile="F:\7.����ѧԺ������\1.�������ֿ�\trains"
dbms=excel replace;
getnames=yes;
mixed=yes;
scantext=yes;
usedate=yes;
scantime=yes;
run;
