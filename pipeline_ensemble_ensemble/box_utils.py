import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import floor

def computemost(a, b, c):
    arr = np.zeros(3)
    arr[a.astype(int)] += 1
    arr[b.astype(int)] += 1
    arr[c.astype(int)] += 1
    if arr[1] >= 2:
        return 1
    elif arr[2] >= 2:
        return 2
    elif arr[1] == 1 and arr[0] == 2:
        return 1
    elif arr[2] == 1 and arr[0] == 2:
        return 2
    else:
        return 0


def checkintersection(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    if ax2 > bx1 and bx2 > ax1 and ay2 > by1 and by2 > ay1:
        return True
    else:
        return False

    
def distinguishing(mask_a, mask_b, box_a, box_b):
    if checkintersection(box_a, box_b) == False:
        print("The two boxes do not intersect!")
    else:
        ax1, ay1, ax2, ay2 = box_a.astype(int)
        bx1, by1, bx2, by2 = box_b.astype(int)

        list1 = [ax1, ax2, bx1, bx2]
        list2 = [ay1, ay2, by1, by2]
        list1.sort()
        list2.sort()

        left = list1[1]
        right = list1[2]
        up = list2[1]
        down = list2[2]

        mask0 = np.zeros((down - up + 1, right - left + 1)) # Record the result of intersection box

        for i in range(up, down + 1):
            for j in range(left, right + 1):
                if mask_a[i - ay1][j - ax1] == 1 and mask_b[i - by1][j - bx1] == 1:
                    mask0[i - up][j - left] = -1
                elif mask_a[i - ay1][j - ax1] == 1 and mask_b[i - by1][j - bx1] == 0:
                    mask0[i - up][j - left] = 1
                elif mask_a[i - ay1][j - ax1] == 0 and mask_b[i - by1][j - bx1] == 1:
                    mask0[i - up][j - left] = 2

        '''Coloring the outer-most layer of the intersection box'''
        if up == by1:
            for i in range(left, right):
                #print(up, ay1, up - ay1)
                #print(i, ax1, i - ax1)
                #print(up, by1, up - by1)
                #print(i, bx1, i - bx1)
                #print(mask_a.shape)
                #print(mask_b.shape)
                #if(mask_a[up - ay1][i - ax1] == 1 and mask_b[up - by1][i - bx1] == 1):
                if mask0[0][i - left] == -1:
                    mask0[0][i - left] = 1
        elif up == ay1:
            for i in range(left, right):
                #if(mask_a[up - ay1][i - ax1] == 1 and mask_b[up - by1][i - bx1] == 1):
                if mask0[0][i - left] == -1:
                    mask0[0][i - left] = 2

        if down == by2:
            for i in range(left, right):
                #if(mask_a[down - ay1][i - ax1] == 1 and mask_b[down - by1][i - bx1] == 1):
                if mask0[down - up][i - left] == -1:
                    mask0[down - up][i - left] = 1
        elif down == ay2:
            for i in range(left, right):
                #print(mask_a.shape)
                #print(mask_b.shape)
                #print(ax1, bx1)
                #print(left, right, right - 1 - ax1)
                #print(down, ay1, by1)
                #if(mask_a[down - ay1][i - ax1] == 1 and mask_b[down - by1][i - bx1] == 1):
                if mask0[down - up][i - left] == -1:
                    mask0[down - up][i - left] = 2

        if left == bx1:
            for i in range(up, down):
                #if(mask_a[i - ay1][left - ax1] == 1 and mask_b[i - by1][left - bx1] == 1):
                if mask0[i - up][0] == -1:
                    mask0[i - up][0] = 1
        elif left == ax1:
            for i in range(up, down):
                #print(i - ay1)
                #print(left - ax1)
                #print(i - by1)
                #print(left - bx1)
                #print(mask_a.shape)
                #print(mask_b.shape)
                #if(mask_a[i - ay1][left - ax1] == 1 and mask_b[i - by1][left - bx1] == 1):
                if mask0[i - up][0] == -1:
                    mask0[i - up][0] = 2

        if right == bx1:
            for i in range(up, down):
                #if(mask_a[i - ay1][right - ax1] == 1 and mask_b[i - by1][right - bx1] == 1):
                if mask0[i - up][right - left] == -1:
                    mask0[i - up][right - left] = 1
        elif right == ax1:
            for i in range(up, down):
                #if(mask_a[i - ay1][right - ax1] == 1 and mask_b[i - by1][right - bx1] == 1):
                if mask0[i - up][right - left] == -1:
                    mask0[i - up][right - left] = 2

        '''Compute mask0'''
        up0 = 1
        down0 = down - up - 1
        left0 = 1
        right0 = right - left - 1

        while(True):
            if up0 >= down0:
                break
            elif left0 >= right0:
                break
            else:
                for i in range(left0, right0):
                    if mask0[up0][i] == -1:
                        mask0[up0][i] = computemost(mask0[up0-1][i-1], mask0[up0-1][i], mask0[up0-1][i+1])
                    if mask0[down0][i] == -1:
                        mask0[down0][i] = computemost(mask0[down0+1][i-1], mask0[down0+1][i], mask0[down0+1][i+1])
                for i in range(up0, down0):
                    if mask0[i][left0] == -1:
                        mask0[i][left0] = computemost(mask0[i-1][left0-1], mask0[i][left0-1], mask0[i+1][left0-1])
                    if mask0[i][right0] == -1:
                        mask0[i][right0] = computemost(mask0[i-1][right0+1], mask0[i][right0+1], mask0[i+1][right0+1])
                
            up0 += 1
            down0 -= 1
            left0 += 1
            right0 -= 1

        '''Modify the value of the two masks based on mask0'''
        for i in range(up, down):
            for j in range(left, right):
                if mask0[i - up][j - left] == 0:
                    mask_a[i - ay1][j - ax1] = 0
                    mask_b[i - by1][j - bx1] = 0
                elif mask0[i - up][j - left] == 1:
                    mask_a[i - ay1][j - ax1] = 1
                    mask_b[i - by1][j - bx1] = 0
                elif mask0[i - up][j - left] == 2:
                    mask_a[i - ay1][j - ax1] = 0
                    mask_b[i - by1][j - bx1] = 1

        return mask_a, mask_b
