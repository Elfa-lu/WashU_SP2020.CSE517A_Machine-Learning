from whoareyou import whoareyou
from genTrainFeatures import genTrainFeatures
from naivebayesCL import naivebayesCL

xTr,yTr = genTrainFeatures()
w,b = naivebayesCL(xTr,yTr)
#whoareyou(w,b)

from naivebayes import naivebayes
x1 = xTr[:,1]
print(x1.shape)
log = naivebayes(xTr, yTr,x1)
print(log)