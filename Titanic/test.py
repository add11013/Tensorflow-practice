#1.
dict1={'banana':42, 'apple':30, 'orange':51}
dict2={i:j for (i,j) in dict1.items() if j>=32}
A=len(dict1)
B=len(dict2)
print(A+B)
#3, 5, 2, 4

#2.
k=55
n=10
def add(k,n):
    k=k+n
add(k,n)
print(k)
#55, 65, 60, 10

#3.
import numpy as np
A=np.array([[1,2],[3,4]])
B=np.array([[3,4],[5,6]])
print(np.dot(A,B))
#[[13 16][29 36]], [[16 13][36 29]], [[3 8][15 24]], [[8 3][24 15]]

#4
pairs=[('sensor_1', 3), ('sensor_4', 4), ('sensor_2', 1),
 ('sensor_3', 2)]
pairs.sort(key=lambda pair:pair[1])
print(pairs[2][0])
#sensor_1, sensor_2, sensor_3, sensor_4

#5
names=['Jack', 'Rick', 'Wendy', 'Frank']
del names[1]
names.reverse()
names.pop(1)
print(names[0])
#Jack, Rick, Wendy, Frank

#6
a='chicken'
b='kitty'
c='red'
string1=a[3:]+b[0:3]+c
print(string1)
#chikited, ckitred, ckenkitred, ckenkittred

#7
n=0
list1=[]
while n<6:
    list1.append(n)
    n=n+1
print(list1[n])
#5, 6, 0, index out of range

#8
sum(list1,1)
#15, 16, 0, 6

