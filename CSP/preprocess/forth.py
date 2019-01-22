#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Tue Jan 22 15:22:38 2019
   Description:
   License: (C)Copyright 2019
'''
#第一题
#  first = input()
# first_Line = first.split(' ')
# n = int(first_Line[0])
# X = int(first_Line[1])
# array_index = []
# array_operation = []
# print(n,X)
# for i in range(n):
#     operation = input()
#     array = operation.split(' ')
#     index = int(array[0])
#     operation_number = int(array[1])
#     array_index.append(index)
#     array_operation.append(operation_number)
# print(array_index)
# print(array_operation)
# for k in range(n):
#     current_index = array_index[n-k-1]
#     current_operation = array_operation[n-k-1]
#     if current_index == 1:
#         X -= current_operation
#     elif current_index == 2:
#         X += current_operation
#     elif current_index == 3:
#         X = X / current_operation
#     else:
#         X = X * current_operation
# print('origin:', int(X))
    

# #第二题
# n = int(input())
# # print('form 0,2 or 4 ,input some numbers:')
# sum = 0 
# #save numbers
# array_number = []
# max_num = 4
# mid_num = 2
# min_num = 0
# number = input()
# array = number.split(' ')
# prefix = 0
# for i in range(n): 
#     array_number.append(int(array[i]))

# output = []
# output.append(0)
# if n == 1:
#     sum = array[0] * array[0]
# else:
#     for k in range(n):
#         if prefix == min_num:
#             if max_num in array_number:
#                 current = max_num
#             elif mid_num in array_number:
#                 current = mid_num
#             else:
#                 current = min_num
#             output.append(current)
#             array_number.remove(current)
#             prefix = current
#         elif prefix == mid_num:
#             if max_num in array_number:
#                 current = max_num
#             elif min_num in array_number:
#                 current = min_num
#             else:
#                 current = mid_num
#             output.append(current)
#             array_number.remove(current)
#             prefix = current
#         else:
#             if min_num in array_number:
#                 current = min_num
#             elif mid_num in array_number:
#                 current = mid_num
#             else:
#                 current = max_num
#             output.append(current)
#             array_number.remove(current)
#             prefix = current
# # print(output)
# for m in range(1,len(output)):
#     sum += (output[m] - output[m-1]) * (output[m] - output[m-1])
    
# print(sum)

#第三题
n = int(input())
string = input()
array = []
numbers = string.split(' ')
for i in range(len(numbers)):
    array.append(int(numbers[i]))

origin = array[0]
end = array[-1]
output = origin ^ end
# print(output)
result = 0
new_array = []
if end >= origin:
    result = -1
else:
    array.sort()
    index1 = array.index(origin)
    index2 = array.index(end)
    gap = index1 - index2 
    if gap == 1:
        result = output
    else:
        new_array = array[index2+1:index1]
origin_array = []
origin_array.append(origin)   
for m in range(len(new_array)):
    for n in range(len(origin_array)):
        process = origin_array[n] ^ new_array[m]
        origin_array.append(process)
output_result = []
for kk in range(len(origin_array)):
    process2 = origin_array[kk] ^ end
    output_result.append(process2)
output_Result = max(output_result)
if output_Result == 0 or end >= origin:
    result = -1
else:
    result = output_Result
print(result)
    
        